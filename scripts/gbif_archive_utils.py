"""Utilities for inspecting and reading GBIF ZIP / Darwin Core-like archives."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Iterable


GBIF_REQUIRED_COLUMNS = {
    "species",
    "decimallatitude",
    "decimallongitude",
    "taxonrank",
    "occurrencestatus",
    "issue",
    "countrycode",
    "stateprovince",
    "year",
}

TEXT_ENCODINGS = ["utf-8-sig", "utf-8", "latin1"]


def list_zip_members(zip_path: Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(zip_path) as archive:
        return archive.infolist()


def _read_sample_bytes(archive: zipfile.ZipFile, member_name: str, n_bytes: int = 65536) -> bytes:
    with archive.open(member_name) as handle:
        return handle.read(n_bytes)


def detect_text_encoding(sample_bytes: bytes) -> str:
    for encoding in TEXT_ENCODINGS:
        try:
            sample_bytes.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin1"


def detect_delimiter(sample_text: str) -> str:
    header = sample_text.splitlines()[0] if sample_text.splitlines() else ""
    candidates = ["\t", ";", ",", "|"]
    counts = {candidate: header.count(candidate) for candidate in candidates}
    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best
    try:
        return csv.Sniffer().sniff(sample_text, delimiters="\t;,|").delimiter
    except csv.Error:
        return "\t"


def sample_member(
    zip_path: Path,
    member_name: str,
    n_lines: int = 5,
) -> dict[str, object]:
    with zipfile.ZipFile(zip_path) as archive:
        sample_bytes = _read_sample_bytes(archive, member_name)
    encoding = detect_text_encoding(sample_bytes)
    sample_text = sample_bytes.decode(encoding, errors="replace")
    delimiter = detect_delimiter(sample_text)
    lines = sample_text.splitlines()[:n_lines]
    header = lines[0].split(delimiter) if lines else []
    normalized_columns = {column.strip().lower() for column in header}
    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "lines": lines,
        "header": header,
        "normalized_columns": normalized_columns,
    }


def _score_member(member_name: str, columns: set[str]) -> tuple[int, int]:
    name = member_name.lower()
    score = 0
    if name.endswith("occurrence.txt"):
        score += 100
    if "occurrence" in name:
        score += 60
    if name.endswith(".txt"):
        score += 20
    if name.endswith(".tsv"):
        score += 18
    if name.endswith(".csv"):
        score += 16
    score += 5 * len(columns & GBIF_REQUIRED_COLUMNS)
    return score, len(columns & GBIF_REQUIRED_COLUMNS)


def detect_occurrence_member(zip_path: Path) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            lower = info.filename.lower()
            if not lower.endswith((".txt", ".csv", ".tsv")):
                continue
            sample_bytes = _read_sample_bytes(archive, info.filename)
            encoding = detect_text_encoding(sample_bytes)
            sample_text = sample_bytes.decode(encoding, errors="replace")
            delimiter = detect_delimiter(sample_text)
            lines = sample_text.splitlines()
            header = lines[0].split(delimiter) if lines else []
            normalized_columns = {column.strip().lower() for column in header}
            score, matched = _score_member(info.filename, normalized_columns)
            candidates.append(
                {
                    "member_name": info.filename,
                    "size": info.file_size,
                    "encoding": encoding,
                    "delimiter": delimiter,
                    "header": header,
                    "normalized_columns": normalized_columns,
                    "score": score,
                    "matched_required_columns": matched,
                }
            )
    if not candidates:
        raise FileNotFoundError(f"No text-like occurrence candidates found inside {zip_path}")
    candidates.sort(
        key=lambda item: (
            int(item["score"]),
            int(item["matched_required_columns"]),
            int(item["size"]),
        ),
        reverse=True,
    )
    return candidates[0]


def delimiter_label(delimiter: str) -> str:
    labels = {"\t": "tab", ";": "semicolon", ",": "comma", "|": "pipe"}
    return labels.get(delimiter, repr(delimiter))


def open_member_text(
    archive: zipfile.ZipFile,
    member_name: str,
    encoding: str,
):
    raw = archive.open(member_name)
    text = io.TextIOWrapper(raw, encoding=encoding, errors="replace", newline="")
    return raw, text


def normalize_stateprovince(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def is_sao_paulo_state(value: object) -> bool:
    state = normalize_stateprovince(value)
    return state in {"sp", "sao paulo", "são paulo"} or "sao paulo" in state or "são paulo" in state

