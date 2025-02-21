# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "duckdb==1.2.0",
#     "marimo[sql]",
#     "numpy==2.2.3",
#     "openmatrix==0.3.5.0",
#     "polars==1.22.0",
#     "pyarrow==19.0.1",
# ]
# ///

import marimo

__generated_with = "0.11.6"
app = marimo.App(width="medium", app_title="Input Converter")


@app.cell
def _():
    import marimo as mo
    import os
    from pathlib import Path
    import itertools
    import numpy as np
    import polars as pl
    import openmatrix as omx
    import pyarrow as pa
    import pyarrow.parquet as pq
    return Path, itertools, mo, np, omx, os, pa, pl, pq


@app.cell
def _(mo):
    mo.md(r"""# Convert CSV and OMX files to the Parquet format""")
    return


@app.cell
def _(mo):
    ui_file_selector = (
        mo.md("""
    {input_files}

    **output directory**: {output_directory}
    """)
        .batch(
            input_files=mo.ui.file_browser(
                initial_path=r"./",
                multiple=True,
                filetypes=[".csv", ".omx"],
            ),
            output_directory=mo.ui.text(""),
        )
        .form()
    )
    return (ui_file_selector,)


@app.cell
def _(ui_file_selector):
    ui_file_selector
    return


@app.cell
def _(convert_omx, os, pl):
    def convert_input(input_file_info, output_directory):
        input_file_extension = os.path.splitext(input_file_info.path)[1]
        input_path = input_file_info.path
        output_path = os.path.join(
            output_directory,
            os.path.splitext(input_file_info.name)[0] + ".parquet",
        )

        if input_file_extension == ".csv":
            input = pl.read_csv(input_path)
            input.write_parquet(output_path)
        elif ".omx":
            convert_omx(input_path, output_path)
        else:
            raise f"Doesn't know how to convert {input_file_extension} :("

        return f"**{input_file_info.name}** -> **{output_path}**"
    return (convert_input,)


@app.cell
def convert_omx(itertools, np, omx, pa, pq):
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
            origins = np.array([t[0] for t in od_pairs], dtype=np.int32)
            destinations = np.array([t[1] for t in od_pairs], dtype=np.int32)
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


    def convert_omx(input_file, output_file) -> None:
        """
        Convert a OMX file to Parquet format.
        Parameters:
        - input_file: str or Path
            The path to the input OMX file.
        - output_file: str or Path
            The path to save the output dataframe file.
        Returns:
        None
        """
        pa_table = create_matrix_table(input_file)
        pq.write_table(
            pa_table, output_file, compression="ZSTD", compression_level=4
        )
    return convert_omx, create_matrix_table, extract_matrix


@app.cell
def _(Path, ui_file_selector):
    input_file_infos = ui_file_selector.value.get("input_files")
    output_directory = ui_file_selector.value.get("output_directory")

    output_directory_exists = Path(output_directory).exists()
    return input_file_infos, output_directory, output_directory_exists


@app.cell
def _(
    convert_input,
    input_file_infos,
    mo,
    output_directory,
    output_directory_exists,
):
    _msgs = []
    if output_directory_exists & len(input_file_infos) > 0:
        for input_file_info in input_file_infos:
            _msgs.append(
                convert_input(input_file_info, output_directory=output_directory)
            )
        _ui_callout = mo.callout(
            mo.vstack(
                [mo.md(_msg) for _msg in ["## ✅ Converted:"] + _msgs],
                gap=0,
                justify="end",
            ),
            kind="success",
        )
    elif len(input_file_infos) == 0:
        _ui_callout = mo.callout(
            mo.md(
                f"## ⚠️ Please select the files you would like to convert to Parquet and click the 'Submit' button."
            ),
            kind="warn",
        )
    else:
        _ui_callout = mo.callout(
            mo.md(
                f"## ⛔️ The output directory (**{output_directory}**) doesn't exist! Click the 'Submit' button to try again."
            ),
            kind="danger",
        )
    _ui_callout
    return (input_file_info,)


if __name__ == "__main__":
    app.run()
