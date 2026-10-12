# Input data

Only `grid_5km.gpkg` is versioned. The other inputs must be obtained and placed here before running the pipeline.

| File | How to obtain |
|---|---|
| `0014792-260430073515954.zip` | GBIF occurrence download, DOI <https://doi.org/10.15468/dl.evgrnx> (GBIF.org, 7 May 2026). Download the "simple" TSV archive from the DOI page and place the ZIP here without unzipping. Filters: Tracheophyta, country Brazil, years 1970–2026, has coordinate, no geospatial issue (4,455,560 records). |
| `mata_atlantica.gpkg` | Atlantic Forest biome boundary (one feature, EPSG:4674). IBGE *Biomas e Sistema Costeiro-Marinho do Brasil* (2019, 1:250,000), as distributed by the `geobr` package: `geobr.read_biomes(year=2019)`, filtered to `code_biome == 4` (Mata Atlântica). |
| `grid_5km.gpkg` | Versioned. 5 km grid over the biome boundary, EPSG:5880, layer `grid_5km`, field `cell_id`. |

Land cover is not stored locally: MapBiomas Collection 10 is queried through Google Earth Engine (asset `projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2`, accessed 2026-08-14).

Step 0 of the pipeline (`scripts/build_tropical_mlm_sequences.py`) reads the GBIF ZIP and the grid and writes `data/processed/occurrences_with_cell_id.csv`, the cohort used by every later step.
