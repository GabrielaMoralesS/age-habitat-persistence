"""Build tropical DAPT sequences from GBIF occurrences grouped by 5 km cell.

Supports reading either:
- a raw GBIF ZIP / Darwin Core-like archive, detecting the primary occurrence
  table automatically, or
- a plain delimited occurrence table.

Outputs:
1. `data/processed/tropical_mlm_sequences.txt`
2. `data/processed/cell_species_summary.csv`
3. `data/processed/occurrences_with_cell_id.csv`
"""

from __future__ import annotations

import argparse
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from gbif_archive_utils import (
    delimiter_label,
    detect_occurrence_member,
    is_sao_paulo_state,
    open_member_text,
)


INPUT_COLUMNS = [
    "species",
    "decimalLatitude",
    "decimalLongitude",
    "taxonRank",
    "occurrenceStatus",
    "issue",
    "countryCode",
    "stateProvince",
    "year",
]

LAT_MIN = -25.7
LAT_MAX = -19.5
LON_MIN = -53.7
LON_MAX = -44.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Filter GBIF occurrences, assign them to the 5 km grid, and export "
            "species-per-cell sequences for tropical DAPT / MLM training."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=root / "data" / "0014792-260430073515954.zip",
        help="GBIF ZIP archive or plain occurrence table.",
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        default=root / "data" / "grid_5km.gpkg",
        help="GeoPackage containing the 5 km grid with a `cell_id` column.",
    )
    parser.add_argument(
        "--grid-layer",
        default="grid_5km",
        help="Layer name inside the GeoPackage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "processed",
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--min-records-per-cell",
        type=int,
        default=10,
        help="Minimum number of filtered occurrence records required per cell.",
    )
    parser.add_argument(
        "--min-species-per-cell",
        type=int,
        default=5,
        help="Minimum number of unique species required per cell.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows per input chunk.",
    )
    parser.add_argument(
        "--predicate",
        choices=["within", "intersects", "contains"],
        default="within",
        help="Spatial join predicate used to assign points to grid cells.",
    )
    parser.add_argument(
        "--csv-separator",
        default=";",
        help="Separator used when reading a plain non-ZIP table.",
    )
    parser.add_argument(
        "--csv-encoding",
        default="latin1",
        help="Encoding used when reading a plain non-ZIP table.",
    )
    return parser.parse_args()


def require_runtime():
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Run this in an environment with `pandas`, "
            "`geopandas`, `shapely`, `pyproj`, and a GeoPackage-capable IO backend."
        ) from exc
    return gpd, pd


def normalize_species(series):
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )
    return cleaned.where(cleaned.ne(""), None)


def print_stats(title: str, stats: Counter) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, label in [
        ("raw_rows", "Rows read"),
        ("after_taxon_rank", "After taxonRank == SPECIES"),
        ("after_species", "After non-empty species"),
        ("after_occurrence_status", "After occurrenceStatus == PRESENT"),
        ("after_numeric_coords", "After numeric coordinates"),
        ("after_nonzero_coords", "After removing 0,0 coordinates"),
        ("after_issue_filter", "After severe issue filter"),
        ("after_sp_filter", "After Sao Paulo filter"),
        ("after_spatial_join", "After 5 km spatial join"),
    ]:
        print(f"{label}: {stats.get(key, 0):,}")


def load_grid(gpd, grid_path: Path, grid_layer: str):
    grid = gpd.read_file(grid_path, layer=grid_layer)[["cell_id", "geometry"]].copy()
    if grid.empty:
        raise SystemExit(f"Grid layer is empty: {grid_path}::{grid_layer}")
    if "cell_id" not in grid.columns:
        raise SystemExit(f"Grid layer has no `cell_id` column: {grid_path}::{grid_layer}")
    if grid.crs is None:
        raise SystemExit(f"Grid layer has no CRS: {grid_path}::{grid_layer}")
    grid["cell_id"] = grid["cell_id"].astype(str).str.strip()
    grid = grid.loc[grid["cell_id"].ne("")].copy()
    return grid


def filter_chunk(chunk, pd, stats: Counter):
    stats["raw_rows"] += len(chunk)
    filtered = chunk.copy()

    for column in INPUT_COLUMNS:
        if column not in filtered.columns:
            filtered[column] = None

    filtered["taxonRank"] = filtered["taxonRank"].fillna("").astype(str).str.strip().str.upper()
    filtered = filtered.loc[filtered["taxonRank"] == "SPECIES"].copy()
    stats["after_taxon_rank"] += len(filtered)

    filtered["species"] = normalize_species(filtered["species"])
    filtered = filtered.loc[filtered["species"].notna()].copy()
    stats["after_species"] += len(filtered)

    filtered["occurrenceStatus"] = (
        filtered["occurrenceStatus"].fillna("").astype(str).str.strip().str.upper()
    )
    filtered = filtered.loc[filtered["occurrenceStatus"] == "PRESENT"].copy()
    stats["after_occurrence_status"] += len(filtered)

    filtered["decimalLatitude"] = pd.to_numeric(filtered["decimalLatitude"], errors="coerce")
    filtered["decimalLongitude"] = pd.to_numeric(filtered["decimalLongitude"], errors="coerce")
    filtered = filtered.loc[
        filtered["decimalLatitude"].notna() & filtered["decimalLongitude"].notna()
    ].copy()
    stats["after_numeric_coords"] += len(filtered)

    filtered = filtered.loc[
        (filtered["decimalLatitude"] != 0) & (filtered["decimalLongitude"] != 0)
    ].copy()
    stats["after_nonzero_coords"] += len(filtered)

    issue_text = filtered["issue"].fillna("").astype(str).str.upper()
    severe_issue = issue_text.str.contains("ZERO_COORDINATE", na=False) | issue_text.str.contains(
        "COUNTRY_COORDINATE_MISMATCH", na=False
    )
    filtered = filtered.loc[~severe_issue].copy()
    stats["after_issue_filter"] += len(filtered)

    state_mask = filtered["stateProvince"].map(is_sao_paulo_state)
    bbox_mask = filtered["decimalLatitude"].between(LAT_MIN, LAT_MAX) & filtered[
        "decimalLongitude"
    ].between(LON_MIN, LON_MAX)
    filtered = filtered.loc[state_mask | bbox_mask].copy()
    stats["after_sp_filter"] += len(filtered)

    return filtered


def iterate_plain_table(path: Path, pd, args: argparse.Namespace):
    return pd.read_csv(
        path,
        sep=args.csv_separator,
        encoding=args.csv_encoding,
        usecols=lambda column: column in INPUT_COLUMNS,
        dtype=str,
        chunksize=args.chunksize,
        low_memory=False,
        on_bad_lines="skip",
    )


def iterate_zip_table(path: Path, pd, args: argparse.Namespace):
    detected = detect_occurrence_member(path)
    member_name = str(detected["member_name"])
    delimiter = str(detected["delimiter"])
    encoding = str(detected["encoding"])

    print("Detected ZIP occurrence source")
    print("-----------------------------")
    print(f"ZIP: {path}")
    print(f"Member: {member_name}")
    print(f"Encoding: {encoding}")
    print(f"Delimiter: {delimiter_label(delimiter)} ({repr(delimiter)})")
    print(f"Matched required columns: {int(detected['matched_required_columns'])}")

    archive = zipfile.ZipFile(path)
    raw_handle, text_handle = open_member_text(archive, member_name, encoding)
    reader = pd.read_csv(
        text_handle,
        sep=delimiter,
        usecols=lambda column: column in INPUT_COLUMNS,
        dtype=str,
        chunksize=args.chunksize,
        low_memory=False,
        on_bad_lines="skip",
    )

    def generator():
        try:
            for chunk in reader:
                yield chunk
        finally:
            text_handle.close()
            raw_handle.close()
            archive.close()

    return generator()


def iterate_input_rows(path: Path, pd, args: argparse.Namespace):
    if path.suffix.lower() == ".zip":
        return iterate_zip_table(path, pd, args)
    print("Detected plain table source")
    print("--------------------------")
    print(f"Path: {path}")
    print(f"Encoding: {args.csv_encoding}")
    print(f"Delimiter: {delimiter_label(args.csv_separator)} ({repr(args.csv_separator)})")
    return iterate_plain_table(path, pd, args)


def build_sequences(args: argparse.Namespace) -> None:
    gpd, pd = require_runtime()

    input_path = Path(args.input_path)
    grid_path = Path(args.grid_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "cell_species_summary.csv"
    sequences_path = output_dir / "tropical_mlm_sequences.txt"
    joined_occurrences_path = output_dir / "occurrences_with_cell_id.csv"

    if not input_path.exists():
        raise SystemExit(f"Missing input path: {input_path}")
    if not grid_path.exists():
        raise SystemExit(f"Missing 5 km grid: {grid_path}")

    grid = load_grid(gpd, grid_path, args.grid_layer)

    stats: Counter = Counter()
    record_counts: Counter = Counter()
    species_by_cell: dict[str, set[str]] = defaultdict(set)
    first_join_write = True

    reader = iterate_input_rows(input_path, pd, args)

    for chunk_number, chunk in enumerate(reader, start=1):
        filtered = filter_chunk(chunk, pd, stats)
        if filtered.empty:
            print(f"Chunk {chunk_number}: no rows left after filters.")
            continue

        points = gpd.GeoDataFrame(
            filtered[
                [
                    "species",
                    "decimalLatitude",
                    "decimalLongitude",
                    "taxonRank",
                    "occurrenceStatus",
                    "issue",
                    "countryCode",
                    "stateProvince",
                    "year",
                ]
            ].copy(),
            geometry=gpd.points_from_xy(filtered["decimalLongitude"], filtered["decimalLatitude"]),
            crs="EPSG:4326",
        ).to_crs(grid.crs)

        joined = gpd.sjoin(points, grid, how="inner", predicate=args.predicate)
        if joined.empty:
            print(f"Chunk {chunk_number}: no rows intersected the 5 km grid.")
            continue

        stats["after_spatial_join"] += len(joined)

        joined_output = joined[
            [
                "species",
                "decimalLatitude",
                "decimalLongitude",
                "taxonRank",
                "occurrenceStatus",
                "issue",
                "countryCode",
                "stateProvince",
                "year",
                "cell_id",
            ]
        ].copy()
        joined_output.to_csv(
            joined_occurrences_path,
            mode="w" if first_join_write else "a",
            header=first_join_write,
            index=False,
            encoding="utf-8",
        )
        first_join_write = False

        for cell_id, species_name in zip(joined["cell_id"].astype(str), joined["species"]):
            record_counts[cell_id] += 1
            species_by_cell[cell_id].add(species_name)

        print(
            f"Chunk {chunk_number}: "
            f"{len(chunk):,} read | {len(filtered):,} after filters | {len(joined):,} joined"
        )

    summary_rows = []
    final_species_pool: set[str] = set()

    for cell_id in sorted(record_counts):
        n_records = record_counts[cell_id]
        species_set = species_by_cell[cell_id]
        n_species = len(species_set)
        if n_records < args.min_records_per_cell or n_species < args.min_species_per_cell:
            continue
        ordered_species = sorted(species_set)
        final_species_pool.update(ordered_species)
        summary_rows.append(
            {
                "cell_id": cell_id,
                "n_records": n_records,
                "n_species": n_species,
                "species_list": ", ".join(ordered_species),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["cell_id", "n_records", "n_species", "species_list"],
    )
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["cell_id"]).reset_index(drop=True)

    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    summary_df["species_list"].to_csv(
        sequences_path,
        index=False,
        header=False,
        encoding="utf-8",
    )
    if first_join_write:
        pd.DataFrame(
            columns=[
                "species",
                "decimalLatitude",
                "decimalLongitude",
                "taxonRank",
                "occurrenceStatus",
                "issue",
                "countryCode",
                "stateProvince",
                "year",
                "cell_id",
            ]
        ).to_csv(joined_occurrences_path, index=False, encoding="utf-8")

    print_stats("Filtering statistics", stats)

    print("\nFinal output")
    print("------------")
    print(f"Cells with >=1 joined record: {len(record_counts):,}")
    print(f"Cells passing thresholds: {len(summary_df):,}")
    print(
        f"Thresholds used: min_records_per_cell >= {args.min_records_per_cell}, "
        f"min_species_per_cell >= {args.min_species_per_cell}"
    )
    print(
        f"Records retained in final cells: "
        f"{int(summary_df['n_records'].sum()) if not summary_df.empty else 0:,}"
    )
    print(f"Unique species retained in final cells: {len(final_species_pool):,}")
    print(f"Summary CSV: {summary_path}")
    print(f"MLM sequences: {sequences_path}")
    print(f"Joined occurrences: {joined_occurrences_path}")


def main() -> None:
    args = parse_args()
    build_sequences(args)


if __name__ == "__main__":
    main()
