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

__generated_with = "0.11.7"
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

    {overwrite_output}
    """)
        .batch(
            input_files=mo.ui.file_browser(
                initial_path=r"./",
                multiple=True,
                filetypes=[".csv", ".omx"],
            ),
            output_directory=mo.ui.text(""),
            overwrite_output=mo.ui.checkbox(
                label="Overwrite files in the output directory"
            ),
        )
        .form()
    )
    return (ui_file_selector,)


@app.cell
def _(ui_file_selector):
    ui_file_selector
    return


@app.cell(hide_code=True)
def _(Path, convert_omx, os, pl, pq):
    def convert_input(input_file_info, output_directory, overwrite_output):
        input_file_extension = os.path.splitext(input_file_info.path)[1]
        input_path = input_file_info.path
        output_path = os.path.join(
            output_directory,
            os.path.splitext(input_file_info.name)[0] + ".parquet",
        )

        if Path(output_path).exists() and overwrite_output is not True:
            return f"\N{cross mark} **{input_file_info.name}** already exists **{output_path}**"

        if input_file_extension == ".csv":
            table = pl.read_csv(input_path)
            table.write_parquet(output_path)
        elif ".omx":
            table = convert_omx(input_path, output_path)
            pq.write_table(
                table, output_path, compression="ZSTD", compression_level=4
            )
        else:
            raise f"Doesn't know how to convert {input_file_extension} :("

        return f"\N{front-facing baby chick} **{input_file_info.name}** -> **{output_path}**"
    return (convert_input,)


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


@app.cell
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
            for input_file_info in mo.status.progress_bar(
                input_file_infos,
                title="\N{hatching chick} Converting",
                subtitle="Please wait",
                show_eta=False,
                show_rate=False,
                remove_on_exit=False,
            ):
                _msgs.append(
                    convert_input(
                        input_file_info,
                        output_directory=output_directory,
                        overwrite_output=overwrite_output,
                    )
                )
                _ui_callout = mo.callout(
                    mo.vstack(
                        [mo.md(_msg) for _msg in ["##  \N{orange heart} Converted:"] + _msgs],
                        gap=0,
                        justify="end",
                    ),
                    kind="success",
                )
        elif len(input_file_infos) == 0:
            _ui_callout = mo.callout(
                mo.md(
                    f"##  \N{raised hand} Please select the files you would like to convert to Parquet and click the 'Submit' button."
                ),
                kind="warn",
            )
        else:
            _ui_callout = mo.callout(
                mo.md(
                    f"## \N{cross mark} The output directory (**{output_directory}**) doesn't exist! Click the 'Submit' button to try again."
                ),
                kind="danger",
            )
    except Exception as e:
        _ui_callout = mo.callout(
            mo.md(f"""## \N{cross mark} Error in converting `{input_file_info.name}`! 

                **Python error message**:

                {e}
                """),
            kind="danger",
        )

    _ui_callout
    return (input_file_info,)


if __name__ == "__main__":
    app.run()
