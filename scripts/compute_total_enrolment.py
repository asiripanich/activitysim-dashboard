# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "polars==1.22.0",
# ]
# ///


import polars as pl

filepath = r"../data/output_2026/final_land_use.parquet"

land_use = pl.read_parquet(filepath)

land_use = land_use.with_columns(
    total_enrolment = pl.col("primary") + pl.col("secondary") + pl.col("univ")
)

land_use.write_parquet(filepath)