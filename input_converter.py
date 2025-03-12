# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "duckdb==1.2.0",
#     "geopandas==1.0.1",
#     "marimo[sql]",
#     "numpy==2.2.3",
#     "openmatrix==0.3.5.0",
#     "polars==1.22.0",
#     "pyarrow==19.0.1",
# ]
# ///

import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium", app_title="Input Converter")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import os
    from pathlib import Path
    import itertools
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    import openmatrix as omx
    import pyarrow as pa
    import pyarrow.parquet as pq
    import geopandas as gpd
    return Path, cs, gpd, itertools, mo, np, omx, os, pa, pl, pq


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(
        rf"""
        # Convert CSV, SHP, GeoJSON, and OMX files to Parquet format
        """
    )
    ,
        mo.callout(
        """
            Warning: For consistency, all zone ID columns will be automatically converted to the `pl.Int64` data type. This applies to any column named 'origin', 'destination', or any column with a name ending in 'zone_id'.
    """
    , kind='warn')
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    ui_initial_path = (
        mo.md("""
        {initial_path}
        """)
        .batch(
            initial_path=mo.ui.text(
                value=str(mo.notebook_location()),
                full_width=True,
                label="### **Input directory**",
            )
        )
        .form()
    )
    return (ui_initial_path,)


@app.cell(hide_code=True)
def _(ui_initial_path):
    ui_initial_path
    return


@app.cell(hide_code=True)
def _(Path, initial_path, mo):
    if not Path(initial_path).is_dir():
        input_path_check = mo.callout(
            mo.md("""
           ## \N{CROSS MARK} The input directory does not exists! Please fix and click submit again.
        """),
            kind="danger",
        )
    else:
        input_path_check = None
    input_path_check
    return (input_path_check,)


@app.cell(hide_code=True)
def _(mo, ui_initial_path):
    initial_path = (
        str(ui_initial_path.value.get("initial_path"))
        if ui_initial_path.value is not None
        else mo.notebook_location()
    )
    return (initial_path,)


@app.cell(hide_code=True)
def ui_file_selector(initial_path, input_path_check, mo):
    input_path_check
    ui_file_selector = (
        mo.md("""
    **Folders and compatible input files in the input directory**:
    {input_files}

    ### **output directory**: 
    {output_directory}

    {overwrite_output}
    """)
        .batch(
            input_files=mo.ui.file_browser(
                initial_path=initial_path,
                multiple=True,
                filetypes=[".csv", ".shp", ".geojson", ".omx"],
            ),
            output_directory=mo.ui.text("", full_width=True),
            overwrite_output=mo.ui.checkbox(
                label="#### Overwrite files in the output directory"
            ),
        )
        .form()
    )
    return (ui_file_selector,)


@app.cell(hide_code=True)
def _(ui_file_selector):
    ui_file_selector
    return


@app.cell(hide_code=True)
def _(Path, convert_omx, convert_shp, cs, os, pl, pq):
    def convert_input(input_file_info, output_directory, overwrite_output):
        input_file_extension = os.path.splitext(input_file_info.path)[1]
        input_path = input_file_info.path
        output_path = os.path.join(
            output_directory,
            os.path.splitext(input_file_info.name)[0] + ".parquet",
        )

        if Path(output_path).exists() and overwrite_output is not True:
            return f"\N{CROSS MARK} **{input_file_info.name}** already exists **{output_path}**"

        if input_file_extension == ".csv":
            table = pl.read_csv(input_path)
            (
                table
                .with_columns(
                    cs.ends_with("zone_id").cast(pl.Int64), 
                    cs.matches('^destination$').cast(pl.Int64), 
                    cs.matches('^origin$').cast(pl.Int64)
                )
                .write_parquet(output_path)
            )
        elif input_file_extension in [".shp", ".geojson"]:
            convert_shp(input_path, output_path)
        elif input_file_extension == ".omx":
            table = convert_omx(input_path)
            pq.write_table(
                table, output_path, compression="ZSTD", compression_level=4
            )
        else:
            raise f"Doesn't know how to convert {input_file_extension} :("

        return f"\N{FRONT-FACING BABY CHICK} **{input_file_info.name}** -> **{output_path}**"
    return (convert_input,)


@app.cell
def _(gpd):
    def convert_shp(input_path, output_path):
        shp = gpd.read_file(input_path)
        shp.to_parquet(output_path)
    return (convert_shp,)


@app.cell(hide_code=True)
def convert_omx(itertools, np, omx, pa):
    def create_matrix_table(omx_path: str) -> pa.Table:
        """
        Convert all matrices information into a consolidated table.
        Parameters:
            omx_path (str): The file path of the OMX file.
        Returns:
            pa.Table: The consolidated table containing all matrices information.
        """
        with omx.open_file(omx_path) as matrix:
            zones = matrix.shape()[1]
            od_pairs = list(
                itertools.product(range(1, zones + 1), range(1, zones + 1))
            )
            origins = np.array([t[0] for t in od_pairs], dtype=np.int64)
            destinations = np.array([t[1] for t in od_pairs], dtype=np.int64)
            table_contents = {
                "origin": origins,
                "destination": destinations,
            }
            for table_name in matrix.list_matrices():
                column = extract_matrix(matrix, table_name)
                table_contents[table_name] = column

        pa_table = pa.table(table_contents)
        return pa_table


    def extract_matrix(matrix: omx.File, name) -> np.ndarray:
        """
        Extract matrix contents as a row-major vector.

        Parameters:
            matrix (omx.File): The matrix object.
            name: The name of the matrix.

        Returns:
            np.ndarray: The matrix contents as a row-major vector.
        """
        return np.reshape(matrix[name], -1).astype(np.float32)


    def convert_omx(input_file) -> pa.Table:
        """
        Convert a OMX file to Parquet format.
        Parameters:
        - input_file: str or Path
            The path to the input OMX file.
        Returns:
        None
        """
        return create_matrix_table(input_file)
    return convert_omx, create_matrix_table, extract_matrix


@app.cell(hide_code=True)
def _(Path, ui_file_selector):
    if ui_file_selector.value is not None:
        input_file_infos = ui_file_selector.value.get("input_files")
        output_directory = ui_file_selector.value.get("output_directory")
        output_directory_exists = Path(output_directory).exists()
        overwrite_output = ui_file_selector.value.get("overwrite_output")
    else:
        input_file_infos = None
        output_directory = None
        output_directory_exists = None
        overwrite_output = None
    return (
        input_file_infos,
        output_directory,
        output_directory_exists,
        overwrite_output,
    )


@app.cell(hide_code=True)
def _(
    convert_input,
    input_file_infos,
    mo,
    output_directory,
    output_directory_exists,
    overwrite_output,
):
    _msgs = []
    try:
        if (
            input_file_infos is None
            and output_directory is None
            and output_directory_exists is None
        ):
            _ui_callout = mo.md("")
        elif output_directory_exists and len(input_file_infos) > 0:
            with mo.status.progress_bar(
                total=len(input_file_infos),
                show_eta=False,
                show_rate=False,
            ) as bar:
                for input_file_info in input_file_infos:
                    bar.update(
                        title=f"\N{HATCHING CHICK} Converting {input_file_info.name}"
                    )
                    _msgs.append(
                        convert_input(
                            input_file_info,
                            output_directory=output_directory,
                            overwrite_output=overwrite_output,
                        )
                    )
                    _ui_callout = mo.callout(
                        mo.vstack(
                            [
                                mo.md(_msg)
                                for _msg in ["##  \N{ORANGE HEART} Converted:"]
                                + _msgs
                            ],
                            gap=0,
                            justify="end",
                        ),
                        kind="success",
                    )
        elif len(input_file_infos) == 0:
            _ui_callout = mo.callout(
                mo.md(
                    f"##  \N{RAISED HAND} Please select the files you would like to convert to Parquet and click the 'Submit' button."
                ),
                kind="warn",
            )
        else:
            _ui_callout = mo.callout(
                mo.md(
                    f"## \N{CROSS MARK} The output directory (**{output_directory}**) doesn't exist! Click the 'Submit' button to try again."
                ),
                kind="danger",
            )
    except Exception as e:
        _ui_callout = mo.callout(
            mo.md(f"""## \N{CROSS MARK} Error in converting `{input_file_info.name}`! 

                **Python error message**:

                {e}
                """),
            kind="danger",
        )

    _ui_callout
    return bar, input_file_info


if __name__ == "__main__":
    app.run()
