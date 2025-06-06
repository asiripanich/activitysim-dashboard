# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "geopandas==1.0.1",
#     "great-tables==0.16.1",
#     "marimo",
#     "pandas==2.2.3",
#     "plotly[express]==6.0.0",
#     "polars==1.22.0",
#     "pyarrow==19.0.0",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full", app_title="ActivitySim dashboard")


@app.cell(hide_code=True)
def import_packages():
    import marimo as mo
    import os
    import polars as pl
    import polars.selectors as cs
    import geopandas as gpd
    from typing import Any, Dict, Optional, Union, List
    import plotly
    import plotly.express as px
    from great_tables import GT, style, loc, md, system_fonts
    import tomllib
    import altair as alt

    pl.enable_string_cache()
    return (
        Any,
        Dict,
        GT,
        List,
        Optional,
        Union,
        alt,
        cs,
        gpd,
        loc,
        md,
        mo,
        os,
        pl,
        plotly,
        px,
        style,
        system_fonts,
        tomllib,
    )


@app.cell(hide_code=True)
def upload_configs(mo):
    upload_configs = mo.ui.file(
        filetypes=[".toml"],
        multiple=False,
        label="### \N{WHITE RIGHT POINTING BACKHAND INDEX} Click to upload your dashboard configuration file",
    )
    upload_configs
    return (upload_configs,)


@app.cell(hide_code=True)
def read_configs(mo, tomllib, upload_configs):
    if mo.running_in_notebook():
        CONFIGS = tomllib.loads(upload_configs.value[0].contents.decode())
    else:
        with open("dashboard-configs.toml", "rb") as _file:
            CONFIGS = tomllib.load(_file)
    MODELS = CONFIGS["models"]
    return CONFIGS, MODELS


@app.cell(hide_code=True)
def ui_title(CONFIGS, mo):
    mo.hstack(
        [mo.md(f"# {CONFIGS['dashboard']['title']}")],
        justify="end",
    )
    return


@app.cell(hide_code=True)
def scenario_colors():
    scenario_discrete_color_map = {"Base": "#bac5c5", "Project": "#119992"}
    return (scenario_discrete_color_map,)


@app.cell(hide_code=True)
def input_settings(INPUT_DIRS_EXIST, mo, ui_folder_settings_form):
    INPUT_DIRS_EXIST

    base_label = mo.ui.text(
        placeholder="Output folder...", label="**Label:** ", value="Base"
    )

    base_dir = ui_folder_settings_form.value.get("base_dir")

    proj_label = mo.ui.text(
        placeholder="Label...", label="**Label:** ", value="Project"
    )

    proj_dir = ui_folder_settings_form.value.get("proj_dir")
    return base_dir, proj_dir


@app.cell(hide_code=True)
def banner_html_code(mo):
    mo.md(
        r"""
    <head>
      <meta charset="UTF-8" />
      <style>
        .take-challenge-btn {
          background: linear-gradient(to right, #bac5c5, #26d0ce);
          border: none;
          border-radius: 4px;
          color: #ffffff;
          padding: 10px 20px;
          min-width: 530px;
          max-height: 60px;
          align-items: center;
        }
      </style>
    </head>
    """
    )
    return


@app.cell(hide_code=True)
def banner(mo, scenario_discrete_color_map):
    ui_banner = mo.hstack(
        [
            mo.md(f"""
                        <button class="take-challenge-btn" style='background: linear-gradient(to right, {scenario_discrete_color_map["Base"]}, #ffffff);'>
                            <h1 style="text-align: left;"> Base </h1> 
                      </button>"""),
            mo.md(f"""
                        <button class="take-challenge-btn" style="background: linear-gradient(to left, {scenario_discrete_color_map["Project"]}, #ffffff); color: #ffffff;">
                            <h1 style="text-align: right;"> Project </h1>
                      </button>"""),
        ],
    )
    return (ui_banner,)


@app.cell(hide_code=True)
def ui_banner(ui_banner):
    ui_banner
    return


@app.cell(hide_code=True)
def ui_folder_settings(mo):
    ui_folder_settings = mo.hstack(
        [mo.md("{base_dir}"), mo.md("{proj_dir}")], widths="equal"
    )
    return (ui_folder_settings,)


@app.cell(hide_code=True)
def ui_folder_settings_form(CONFIGS, mo, ui_folder_settings):
    ui_folder_settings_form = ui_folder_settings.batch(
        base_dir=mo.ui.text(
            placeholder="Output folder...",
            label="**Base Folder** ",
            value=rf"{CONFIGS['base']['directory']}",
            full_width=True,
        ),
        proj_dir=mo.ui.text(
            placeholder="Output folder...",
            label="**Project Folder** ",
            value=rf"{CONFIGS['project']['directory']}",
            full_width=True,
        ),
    )
    if mo.running_in_notebook():
        ui_folder_settings_form = ui_folder_settings_form.form()
    return (ui_folder_settings_form,)


@app.cell
def ui_folder_settings_form_display(ui_folder_settings_form):
    ui_folder_settings_form
    return


@app.cell(hide_code=True)
def stop_if_form_not_submitted(mo, ui_folder_settings_form):
    """
    Stop execution if the folder settings form has not been submitted.
    """

    mo.stop(
        ui_folder_settings_form.value is None,
        mo.md("**Submit the form to continue**"),
    )
    CHECK_INPUT_DIRS = True
    return (CHECK_INPUT_DIRS,)


@app.cell(hide_code=True)
def check_input_dirs(Any, CHECK_INPUT_DIRS, mo, os, ui_folder_settings_form):
    CHECK_INPUT_DIRS

    _form_vals = ui_folder_settings_form.value


    def _check_dir(path: str, label: str) -> Any:
        if not os.path.exists(path):
            return mo.callout(
                mo.md(f"## \N{CROSS MARK} **{label} folder** was not found."),
                kind="danger",
            )
        return True


    _base_check = _check_dir(_form_vals.get("base_dir"), "Base")
    _proj_check = _check_dir(_form_vals.get("proj_dir"), "Project")
    mo.stop(_base_check is not True, _base_check)
    mo.stop(_proj_check is not True, _proj_check)

    INPUT_DIRS_EXIST = True
    return (INPUT_DIRS_EXIST,)


@app.cell(hide_code=True)
def _():
    ACTIVITYSIM_OUTPUT_FILES = {
        "persons": {"filename": "final_persons.parquet", "required": True},
        "households": {"filename": "final_households.parquet", "required": True},
        "trips": {"filename": "final_trips.parquet", "required": True},
        "tours": {"filename": "final_tours.parquet", "required": True},
        "joint_tour_participants": {
            "filename": "final_joint_tour_participants.parquet",
            "required": True,
        },
        "land_use": {"filename": "final_land_use.parquet", "required": True},
        "skims": {"filename": "skims.parquet", "required": True},
        "zones": {"filename": "zones.parquet", "required": False},
    }
    return (ACTIVITYSIM_OUTPUT_FILES,)


@app.cell(hide_code=True)
def read_input_parquets(
    ACTIVITYSIM_OUTPUT_FILES,
    Any,
    INPUT_DIRS_EXIST,
    Optional,
    base_dir,
    gpd,
    os,
    pl,
    proj_dir,
):
    INPUT_DIRS_EXIST


    def _read_asim_output(asim_path, attributes) -> Optional[Any]:
        filepath = os.path.join(asim_path, attributes["filename"])

        if not os.path.exists(filepath) and not attributes["required"]:
            return None

        if attributes["filename"] == "zones.parquet":
            return gpd.read_parquet(filepath)

        if attributes["filename"] in [
            "final_tours.parquet",
            "final_trips.parquet",
            "skims.parquet",
        ]:
            return pl.scan_parquet(filepath).cast(
                {"origin": pl.Int64, "destination": pl.Int64}
            )

        return pl.scan_parquet(filepath)


    BASE_OUTPUTS = {
        name: _read_asim_output(base_dir, attrs)
        for name, attrs in ACTIVITYSIM_OUTPUT_FILES.items()
        if os.path.exists(os.path.join(base_dir, attrs["filename"]))
    }

    BASE_OUTPUTS['tours'] = BASE_OUTPUTS['tours'].join(BASE_OUTPUTS['persons'], on="person_id")
    BASE_OUTPUTS['trips'] = BASE_OUTPUTS['trips'].join(BASE_OUTPUTS['persons'], on="person_id")

    PROJ_OUTPUTS = {
        name: _read_asim_output(proj_dir, attrs)
        for name, attrs in ACTIVITYSIM_OUTPUT_FILES.items()
    }

    PROJ_OUTPUTS['tours'] = PROJ_OUTPUTS['tours'].join(PROJ_OUTPUTS['persons'], on="person_id")
    PROJ_OUTPUTS['trips'] = PROJ_OUTPUTS['trips'].join(PROJ_OUTPUTS['persons'], on="person_id")

    return BASE_OUTPUTS, PROJ_OUTPUTS


@app.cell
def ui_overview(INPUT_DIRS_EXIST, mo):
    mo.hstack(
        [mo.md(rf"""# {mo.icon("lucide:chart-bar-big")} Overview""")],
        justify="start",
    ) if INPUT_DIRS_EXIST is True else None
    return


@app.cell
def ui_summary_cards(summary_cards):
    summary_cards
    return


@app.cell(hide_code=True)
def summary_cards(
    Any,
    BASE_OUTPUTS,
    CONFIGS,
    Dict,
    Optional,
    PROJ_OUTPUTS,
    mo,
    pl,
):
    # Utility Functions
    def _get_direction(proj_value: float, base_value: float) -> Optional[str]:
        """
        Compare the projected and base values and return a direction string.

        Args:
            proj_value (float): The projected value.
            base_value (float): The baseline value.

        Returns:
            Optional[str]: "increase" if proj_value > base_value,
                           "decrease" if proj_value < base_value,
                           None if the values are equal.
        """
        if proj_value > base_value:
            return "increase"
        elif proj_value < base_value:
            return "decrease"
        return None


    def count_rows(lazy_frame: pl.LazyFrame, weight_col: str = None) -> int:
        """
        Return the total number of rows in a LazyFrame.

        Args:
            lazy_frame (pl.LazyFrame): The LazyFrame to count rows.

        Returns:
            int: The total number of rows.
        """
        if weight_col is not None:
            return lazy_frame.select(pl.col(weight_col).sum()).collect().item()
        return lazy_frame.select(pl.len()).collect().item()


    def count_true(lazy_frame: pl.LazyFrame, column: str) -> Optional[int]:
        """
        Count the number of rows where the given column is True.

        Args:
            lazy_frame (pl.LazyFrame): The LazyFrame to filter.
            column (str): The column to evaluate.

        Returns:
            Optional[int]: The count of rows where the column is True, or None if an error occurs.
        """
        try:
            return (
                lazy_frame.filter(pl.col(column)).select(pl.len()).collect().item()
            )
        except Exception:
            return None


    def count_category(
        lazy_frame: pl.LazyFrame, column: str, category: Any
    ) -> Optional[int]:
        """
        Count the number of rows where the given column equals the specified category.

        Args:
            lazy_frame (pl.LazyFrame): The LazyFrame to filter.
            column (str): The column to evaluate.
            category (Any): The category value to count.

        Returns:
            Optional[int]: The count of rows for the category, or None if an error occurs.
        """
        try:
            return (
                lazy_frame.filter(pl.col(column) == category)
                .select(pl.len())
                .collect()
                .item()
            )
        except Exception:
            return None


    # Summary Computation
    def _compute_summary(
        outputs: Dict[str, pl.LazyFrame], configs: Dict
    ) -> Dict[str, float]:
        """
        Compute summary metrics from a collection of LazyFrames.

        The expected keys in `outputs` are "persons", "households", "trips", and "tours".

        Args:
            outputs (Dict[str, pl.LazyFrame]): Dictionary containing LazyFrames.

        Returns:
            Dict[str, float]: Computed summary metrics.
        """
        total_persons = count_rows(outputs["persons"], configs.get("weight"))
        total_households = count_rows(outputs["households"], configs.get("weight"))
        total_trips = count_rows(outputs["trips"], configs.get("weight"))
        total_tours = count_rows(outputs["tours"], configs.get("weight"))

        return {
            "total_persons": total_persons,
            "total_households": total_households,
            "average_household_size": total_persons / total_households
            if total_households
            else 0,
            "total_trips": total_trips,
            "total_tours": total_tours,
            "avg._trips_per_person": total_trips / total_persons
            if total_persons
            else 0,
            "avg._tours_per_person": total_tours / total_persons
            if total_persons
            else 0,
            # "remote_workers": count_true(
            #     outputs["persons"],
            #     MODELS["household_person"]["work_from_home"]["result_field"],
            # ),
            # "free_parking_at_work": count_true(
            #     outputs["persons"],
            #     MODELS["household_person"]["free_parking_at_work"]["result_field"],
            # ),
            # "zero_car_households": count_category(
            #     outputs["households"],
            #     MODELS["household_person"]["auto_ownership"]["result_field"],
            #     0,
            # ),
        }


    # Card Production Functions
    def _format_caption(
        proj_value: Optional[float], base_value: Optional[float]
    ) -> Optional[str]:
        """
        Format a caption string that shows the percentage difference and base value.

        Args:
            proj_value (Optional[float]): The projected value.
            base_value (Optional[float]): The baseline value.

        Returns:
            Optional[str]: The formatted caption or None if inputs are invalid.
        """
        if proj_value is None or base_value is None:
            return None

        pct_diff = ((proj_value / base_value) - 1) * 100
        decimals = 0 if base_value == int(base_value) else 2
        return f"{pct_diff:.1f}% (Base: {base_value:,.{decimals}f})"


    def _produce_card(
        base_value: Optional[float], proj_value: Optional[float], label: str
    ) -> Any:
        """
        Create a summary card based on the base and projected values.

        Args:
            base_value (Optional[float]): The baseline value.
            proj_value (Optional[float]): The projected value.
            label (str): The label for the card.

        Returns:
            Any: A card object created by the m⊙statmo.stat function.
        """
        formatted_label = label.replace("_", " ").upper()
        caption = _format_caption(proj_value, base_value)
        direction = (
            _get_direction(proj_value, base_value)
            if proj_value is not None and base_value is not None
            else None
        )

        return mo.stat(
            label=formatted_label,
            value=proj_value,
            caption=caption,
            bordered=True,
            direction=direction,
        )


    def generate_summary_cards(summary: Dict[str, Dict[str, float]]) -> Any:
        """
        Generate a vertically stacked container of summary cards from metric dictionaries.

        The ∑marysummary dictionary should have two keys: "base" and "proj", each mapping
        to a dictionary of computed metrics.

        Args:
            summary (Dict[str, Dict[str, float]]): Summary metrics for base and projected values.

        Returns:
            Any: A container object with the summary cards.
        """
        cards = [
            _produce_card(summary["base"].get(key), summary["proj"].get(key), key)
            for key in summary["base"]
        ]
        return mo.vstack(
            [mo.hstack(cards, justify="center", align="center", wrap=True)]
        )


    summary_cards = generate_summary_cards(
        {
            "base": _compute_summary(BASE_OUTPUTS, CONFIGS["base"]),
            "proj": _compute_summary(PROJ_OUTPUTS, CONFIGS["project"]),
        }
    )
    return (summary_cards,)


@app.cell(hide_code=True)
def ui_models(INPUT_DIRS_EXIST, mo):
    mo.hstack(
        [mo.md(rf"""# {mo.icon("lucide:square-chevron-right")} Models""")],
        justify="start",
    ) if INPUT_DIRS_EXIST is True else None
    return


@app.cell(hide_code=True)
def ui_models_helper(INPUT_DIRS_EXIST, mo):
    mo.accordion(
        {
            "#### 👀 Model structure": mo.hstack(
                [
                    mo.image(
                        "https://rsginc.com/wp-content/uploads/2021/01/Example-Activity-Based-Model-and-Submodel-Structure-2048x1907.png",
                        width=500,
                        caption="ActivitySim model structure",
                    )
                ],
                justify="center",
            ),
            # "#### 🔎 Select a variable from the table to group by": column_table,
        }
    ) if INPUT_DIRS_EXIST is True else None
    return


@app.cell(hide_code=True)
def column_filter_table(Any, BASE_OUTPUTS, Dict, List, Union, gpd, pl):
    # Constants with underscore prefix to indicate internal usage
    _EXCLUDE_TABLES = {"skims", "land_use"}
    _COLUMN_SUFFIX_TO_FILTER = "_id"


    def _get_frame_columns(
        frame: Union[gpd.GeoDataFrame, pl.LazyFrame],
    ) -> List[str]:
        """
        Retrieve the list of column names from a GeoDataFrame or LazyFrame.

        Args:
            frame: A GeoDataFrame or LazyFrame object.

        Returns:
            A list of column names as strings.

        Raises:
            TypeError: If the frame type is not supported.
        """
        if isinstance(frame, gpd.GeoDataFrame):
            return list(frame.columns)
        elif isinstance(frame, pl.LazyFrame):
            return list(frame.collect_schema().keys())
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")


    def _get_frame_dtypes(
        frame: Union[gpd.GeoDataFrame, pl.LazyFrame],
    ) -> List[str]:
        """
        Retrieve the data types (as strings) from a GeoDataFrame or LazyFrame.

        Args:
            frame: A GeoDataFrame or LazyFrame object.

        Returns:
            A list of data types as strings.

        Raises:
            TypeError: If the frame type is not supported.
        """
        if isinstance(frame, gpd.GeoDataFrame):
            return [str(dtype) for dtype in frame.dtypes]
        elif isinstance(frame, pl.LazyFrame):
            return [str(dtype) for dtype in frame.collect_schema().dtypes()]
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")


    def _build_columns_dataframe(base_outputs: Dict[str, Any]) -> pl.DataFrame:
        """
        Create a DataFrame that consolidates table names, their columns, and data types,
        excluding any tables specified in _EXCLUDE_TABLES.

        Args:
            base_outputs: A dictionary mapping table names to GeoDataFrame or LazyFrame objects.

        Returns:
            A polars DataFrame with columns: 'table', 'column', and 'data_type'.
        """
        _dfs = []
        for table_name, frame in base_outputs.items():
            if table_name in _EXCLUDE_TABLES:
                continue

            columns = _get_frame_columns(frame)
            dtypes = _get_frame_dtypes(frame)

            # Create a DataFrame for each table where each row represents a column of the table
            table_df = pl.DataFrame(
                {
                    "table": [table_name] * len(columns),
                    "column": columns,
                    "data_type": dtypes,
                }
            )
            _dfs.append(table_df)

        if _dfs:
            return pl.concat(_dfs)
        return pl.DataFrame({"table": [], "column": [], "data_type": []})


    def _filter_columns(
        df: pl.DataFrame, suffix: str = _COLUMN_SUFFIX_TO_FILTER
    ) -> pl.DataFrame:
        """
        Filter out rows from the DataFrame where the 'column' name ends with a given suffix.

        Args:
            df: A polars DataFrame containing a 'column' field.
            suffix: A string suffix to filter out (default is "_id").

        Returns:
            A filtered polars DataFrame.
        """
        return df.filter(~pl.col("column").str.ends_with(suffix))


    def _build_model_result_filter_options(df: pl.DataFrame) -> List[str]:
        """
        Build a list of formatted string options from the DataFrame that include the table name,
        data type, and column name.

        Args:
            df: A polars DataFrame with 'table', 'column', and 'data_type' fields.

        Returns:
            A list of formatted string options.
        """
        _table_list = df["table"].to_list()
        _column_list = df["column"].to_list()
        _dtype_list = df["data_type"].to_list()

        return [
            f"{_table_list[i]} ({_dtype_list[i]}): {_column_list[i]}"
            for i in range(len(_column_list))
        ]


    _all_columns_df = _build_columns_dataframe(BASE_OUTPUTS)
    _filtered_columns_df = _filter_columns(
        _all_columns_df, _COLUMN_SUFFIX_TO_FILTER
    )
    MODEL_RESULT_FILTER_OPTIONS = _build_model_result_filter_options(
        _filtered_columns_df
    )
    return (MODEL_RESULT_FILTER_OPTIONS,)


@app.cell(hide_code=True)
def _(MODEL_RESULT_FILTER_OPTIONS, mo):
    multiselect = mo.ui.multiselect(
        options=MODEL_RESULT_FILTER_OPTIONS,
        label="### \N{PINCHING HAND} Select variables for cross-tabulating the model results",
        full_width=True,
    ).form()
    return (multiselect,)


@app.cell(hide_code=True)
def _(mo, multiselect):
    mo.hstack(
        [
            multiselect,
            mo.md(
                f"**Currect selections**: <br> {'<br> '.join(multiselect.value) if multiselect.value is not None else None}"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def filter_columns(multiselect):
    FILTER_COLUMNS = (
        [x.split(": ")[1] for x in multiselect.value]  # strip the table name
        if multiselect.value is not None
        else []
    )
    return (FILTER_COLUMNS,)


@app.cell(hide_code=True)
def assemble_model_diagnostics(
    BASE_OUTPUTS,
    FILTER_COLUMNS,
    List,
    Optional,
    PROJ_OUTPUTS,
    generate_general_model_diagnostic,
    generate_location_model_diagnostic,
    mo,
    pl,
):
    def assemble_model_diagnostics(model_name, fields, with_ui=True):
        table_name = fields["table"]
        base_lazy_df = BASE_OUTPUTS[table_name]
        proj_lazy_df = PROJ_OUTPUTS[table_name]
        filter_expr = fields.get("filter_expr")

        if filter_expr is not None:
            base_lazy_df = base_lazy_df.sql(filter_expr)
            proj_lazy_df = proj_lazy_df.sql(filter_expr)

        # Ensure result_field is always a list for iteration.
        result_fields = fields["result_field"]
        if not isinstance(result_fields, list):
            result_fields = [result_fields]

        diagnostics = {
            result_field: generate_model_diagnostic(
                base_lazy_df,
                proj_lazy_df,
                result_field,
                fields,
                FILTER_COLUMNS,
                model_name,
            )
            for result_field in result_fields
        }

        if with_ui:
            diagnostics = {
                key: mo.ui.tabs(value) for key, value in diagnostics.items()
            }

        if with_ui and len(result_fields) > 1:
            diagnostics = mo.ui.tabs(diagnostics)

        # Return a single diagnostic if only one, else as a mo.ui.tabs.
        return (
            diagnostics[result_fields[0]]
            if len(result_fields) == 1
            else diagnostics
        )


    def generate_model_diagnostic(
        base_lazy_df: pl.LazyFrame,
        proj_lazy_df: Optional[pl.LazyFrame],
        variable: str,
        fields,
        by_columns: Optional[List[str]] = None,
        model_name: str = None,
    ):
        if model_name.endswith(("_location", "_destination")):
            return generate_location_model_diagnostic(
                base_lazy_df,
                proj_lazy_df,
                variable,
                fields,
                by_columns,
                model_name,
            )
        else:
            return generate_general_model_diagnostic(
                base_lazy_df, proj_lazy_df, variable, by_columns, model_name
            )


    def check_exists(fields):
        table_cols = BASE_OUTPUTS[fields["table"]].collect_schema().keys()
        result_field = fields["result_field"]

        if isinstance(result_field, str):
            return result_field in table_cols

        return all(item in table_cols for item in result_field)
    return assemble_model_diagnostics, check_exists


@app.cell(hide_code=True)
def _(INPUT_DIRS_EXIST, mo):
    mo.md("""### Households/Persons""") if INPUT_DIRS_EXIST is True else None
    return


@app.cell(hide_code=True)
def _(MODELS, assemble_model_diagnostics, check_exists):
    main_accordion = {"household_person": {}, "tour": {}, "trip": {}}

    results = {}
    for _tab_name in main_accordion.keys():
        for _model_name, _fields in MODELS.get(_tab_name).items():
            # print({_model_name})
            if check_exists(_fields):
                results[_model_name] = assemble_model_diagnostics(
                    _model_name, _fields, with_ui=False
                )
    return (results,)


@app.cell(hide_code=True)
def auto_ownership(generate_model_ui):
    generate_model_ui("auto_ownership")
    return


@app.cell(hide_code=True)
def work_from_home(generate_model_ui):
    generate_model_ui("work_from_home")
    return


@app.cell(hide_code=True)
def free_parking_at_work(generate_model_ui):
    generate_model_ui("free_parking_at_work")
    return


@app.cell(hide_code=True)
def school_location_results(get_location_results_options_configs):
    (
        school_location_results,
        school_location_options,
        school_location_configs,
    ) = get_location_results_options_configs("household_person", "school_location")
    return (
        school_location_configs,
        school_location_options,
        school_location_results,
    )


@app.cell(hide_code=True)
def school_location(
    generate_location_model_ui,
    school_location_configs,
    school_location_options,
    school_location_results,
):
    generate_location_model_ui(
        school_location_results, school_location_configs, school_location_options
    )
    return


@app.cell(hide_code=True)
def work_location_results(get_location_results_options_configs):
    (
        work_location_results,
        work_location_options,
        work_location_configs,
    ) = get_location_results_options_configs("household_person", "work_location")
    return work_location_configs, work_location_options, work_location_results


@app.cell(hide_code=True)
def work_location(
    generate_location_model_ui,
    work_location_configs,
    work_location_options,
    work_location_results,
):
    generate_location_model_ui(
        work_location_results, work_location_configs, work_location_options
    )
    return


@app.cell(hide_code=True)
def business_location_results(get_location_results_options_configs):
    (
        business_location_results,
        business_location_options,
        business_location_configs,
    ) = get_location_results_options_configs(
        "household_person", "business_location"
    )
    return (
        business_location_configs,
        business_location_options,
        business_location_results,
    )


@app.cell(hide_code=True)
def business_location(
    business_location_configs,
    business_location_options,
    business_location_results,
    generate_location_model_ui,
):
    generate_location_model_ui(
        business_location_results,
        business_location_configs,
        business_location_options,
    )
    return


@app.cell(hide_code=True)
def telecommute_frequency(generate_model_ui):
    generate_model_ui("telecommute_frequency")
    return


@app.cell(hide_code=True)
def cdap_simulate(generate_model_ui):
    generate_model_ui("cdap_simulate")
    return


@app.cell(hide_code=True)
def ui_models_tour_section(INPUT_DIRS_EXIST, mo):
    mo.md("""### Tours""") if INPUT_DIRS_EXIST is True else None
    return


@app.cell(hide_code=True)
def mandatory_tour_frequency(generate_model_ui):
    generate_model_ui("mandatory_tour_frequency")
    return


@app.cell(hide_code=True)
def mandatory_tour_scheduling(generate_model_ui):
    generate_model_ui("mandatory_tour_scheduling")
    return


@app.cell(hide_code=True)
def joint_tour_frequency(generate_model_ui):
    generate_model_ui("joint_tour_frequency")
    return


@app.cell(hide_code=True)
def joint_tour_composition(generate_model_ui):
    generate_model_ui("joint_tour_composition")
    return


@app.cell(hide_code=True)
def joint_tour_participation(generate_model_ui):
    generate_model_ui("joint_tour_participation")
    return


@app.cell(hide_code=True)
def joint_tour_destination(
    generate_location_model_ui,
    joint_tour_destination_configs,
    joint_tour_destination_options,
    joint_tour_destination_results,
):
    generate_location_model_ui(
        joint_tour_destination_results,
        joint_tour_destination_configs,
        joint_tour_destination_options,
    )
    return


@app.cell(hide_code=True)
def joint_tour_destination_results(get_location_results_options_configs):
    (
        joint_tour_destination_results,
        joint_tour_destination_options,
        joint_tour_destination_configs,
    ) = get_location_results_options_configs("tour", "joint_tour_destination")
    return (
        joint_tour_destination_configs,
        joint_tour_destination_options,
        joint_tour_destination_results,
    )


@app.cell(hide_code=True)
def joint_tour_scheduling(generate_model_ui):
    generate_model_ui("joint_tour_scheduling")
    return


@app.cell(hide_code=True)
def non_mandatory_tour_frequency(generate_model_ui):
    generate_model_ui("non_mandatory_tour_frequency")
    return


@app.cell(hide_code=True)
def non_mandatory_tour_destination(
    generate_location_model_ui,
    non_mandatory_tour_destination_configs,
    non_mandatory_tour_destination_options,
    non_mandatory_tour_destination_results,
):
    generate_location_model_ui(
        non_mandatory_tour_destination_results,
        non_mandatory_tour_destination_configs,
        non_mandatory_tour_destination_options,
    )
    return


@app.cell(hide_code=True)
def non_mandatory_tour_destination_results(
    get_location_results_options_configs,
):
    (
        non_mandatory_tour_destination_results,
        non_mandatory_tour_destination_options,
        non_mandatory_tour_destination_configs,
    ) = get_location_results_options_configs(
        "tour", "non_mandatory_tour_destination"
    )
    return (
        non_mandatory_tour_destination_configs,
        non_mandatory_tour_destination_options,
        non_mandatory_tour_destination_results,
    )


@app.cell(hide_code=True)
def non_mandatory_tour_scheduling(generate_model_ui):
    generate_model_ui("non_mandatory_tour_scheduling")
    return


@app.cell(hide_code=True)
def tour_mode_choice_simulate(generate_model_ui):
    generate_model_ui("tour_mode_choice")
    return


@app.cell(hide_code=True)
def atwork_subtour_frequency(generate_model_ui):
    generate_model_ui("atwork_subtour_frequency")
    return


@app.cell(hide_code=True)
def atwork_subtour_destination(
    atwork_subtour_destination_configs,
    atwork_subtour_destination_options,
    atwork_subtour_destination_results,
    generate_location_model_ui,
):
    generate_location_model_ui(
        atwork_subtour_destination_results,
        atwork_subtour_destination_configs,
        atwork_subtour_destination_options,
    )
    return


@app.cell(hide_code=True)
def atwork_subtour_destination_results(get_location_results_options_configs):
    (
        atwork_subtour_destination_results,
        atwork_subtour_destination_options,
        atwork_subtour_destination_configs,
    ) = get_location_results_options_configs("tour", "atwork_subtour_destination")
    return (
        atwork_subtour_destination_configs,
        atwork_subtour_destination_options,
        atwork_subtour_destination_results,
    )


@app.cell(hide_code=True)
def atwork_subtour_scheduling(generate_model_ui):
    generate_model_ui("atwork_subtour_scheduling")
    return


@app.cell(hide_code=True)
def atwork_subtour_mode_choice(generate_model_ui):
    generate_model_ui("atwork_subtour_mode_choice")
    return


@app.cell(hide_code=True)
def stop_frequency(generate_model_ui):
    generate_model_ui("stop_frequency")
    return


@app.cell(hide_code=True)
def ui_models_trip_section(INPUT_DIRS_EXIST, mo):
    mo.md("""### Trips""") if INPUT_DIRS_EXIST is True else None
    return


@app.cell(hide_code=True)
def trip_purpose(generate_model_ui):
    generate_model_ui("trip_purpose")
    return


@app.cell(hide_code=True)
def trip_destination_results(get_location_results_options_configs):
    (
        trip_destination_results,
        trip_destination_options,
        trip_destination_configs,
    ) = get_location_results_options_configs("trip", "trip_destination")
    return (
        trip_destination_configs,
        trip_destination_options,
        trip_destination_results,
    )


@app.cell(hide_code=True)
def trip_destination(
    generate_location_model_ui,
    trip_destination_configs,
    trip_destination_options,
    trip_destination_results,
):
    generate_location_model_ui(
        trip_destination_results,
        trip_destination_configs,
        trip_destination_options,
    )
    return


@app.cell(hide_code=True)
def trip_scheduling_choice(generate_model_ui):
    generate_model_ui("trip_scheduling_choice")
    return


@app.cell(hide_code=True)
def trip_departure_choice(generate_model_ui):
    generate_model_ui("trip_departure_choice")
    return


@app.cell(hide_code=True)
def _(
    CONFIGS,
    FILTER_COLUMNS,
    List,
    MODELS,
    alt,
    assemble_model_diagnostics,
    compute_difference_metrics,
    generate_gt_table,
    mo,
    pl,
    scenario_discrete_color_map,
):
    def recalculate_proportions(
        df: pl.DataFrame, grouping_columns: List[str], skims_variable: str, options
    ):
        if grouping_columns:
            grouping_columns = [
                col for col in grouping_columns if col in df.columns
            ]
        df_recalculated = (
            df.filter(
                pl.col(skims_variable) >= options.value.get("range_slider")[0]
            )
            .with_columns(
                pl.col(skims_variable).cut(
                    range(
                        0,
                        options.value.get("range_slider")[1],
                        options.get("cut").value,
                    )
                )
            )
            .group_by([skims_variable] + grouping_columns)
            .agg(pl.col("count_Base", "count_Project").sum())
            .with_columns(
                share_Base=(
                    pl.col("count_Base") / pl.col("count_Base").sum()
                ).over(grouping_columns if grouping_columns else None),
                share_Project=(
                    pl.col("count_Project") / pl.col("count_Project").sum()
                ).over(grouping_columns if grouping_columns else None),
            )
        )
        return df_recalculated


    def _generate_location_model_chart_ui(
        start=0,
        stop=None,
        default_range=None,
        cut_start=2,
        cut_stop=10,
        cut_default=5,
    ):
        assert stop is not None
        assert default_range is not None
        return mo.md("X-axis range: {range_slider} cut: {cut}").batch(
            range_slider=mo.ui.range_slider(
                start=start,
                stop=stop,
                value=default_range,
                full_width=False,
                show_value=True,
            ),
            cut=mo.ui.slider(
                start=cut_start, stop=cut_stop, value=cut_default, show_value=True
            ),
        )


    def generate_location_model_chart_ui(df, skims_variable):
        return _generate_location_model_chart_ui(
            stop=round(df[skims_variable].max()),
            default_range=[0, round(df[skims_variable].median())],
        )


    def plot_distance_comparison(df, grouping_columns, skims_variable):
        if grouping_columns:
            grouping_columns = [
                col for col in grouping_columns if col in df.columns
            ]
        facet_col = grouping_columns[0] if grouping_columns else None
        base = alt.Chart(df).mark_area(
            line={"color": scenario_discrete_color_map.get("Base")},
            point=False,
            color=scenario_discrete_color_map.get("Project"),
            interpolate="step",
            opacity=0.8,
        )

        chart = base.encode(
            alt.X(skims_variable),
            alt.Y("share_Base:Q").axis(format=".1%").title("Share"),
            alt.Y2("share_Project:Q"),
            # fill=alt.when(alt.datum.share_Base>0.2).then(alt.value("red")).otherwise(alt.value("red")),
        )

        if facet_col is not None:
            columns = min(df[facet_col].n_unique(), 3)
            one_facet_width = 900 / columns
            chart = chart.properties(width=one_facet_width).facet(
                facet=f"{facet_col}:N", columns=columns
            )
        return chart.configure_axis(
            labelFontSize=CONFIGS.get("dashboard").get("axis_labelFontSize"),
            titleFontSize=CONFIGS.get("dashboard").get("axis_titleFontSize"),
        )


    def get_location_results_options_configs(table, model_name):
        configs = MODELS.get(table).get(model_name)
        configs["model_name"] = model_name
        results = assemble_model_diagnostics(model_name, configs, with_ui=False)
        options = generate_location_model_chart_ui(
            results.get("data"),
            configs.get("skims_variable"),
        )
        return results, options, configs


    def generate_location_model_ui(results, configs, options):
        df = recalculate_proportions(
            results.get("data"),
            FILTER_COLUMNS,
            configs.get("skims_variable"),
            options,
        ).sort([configs.get("skims_variable")])

        plot = plot_distance_comparison(
            df, FILTER_COLUMNS, configs.get("skims_variable")
        )

        df_with_diff = df.with_columns(
            diff=pl.col("count_Project") - pl.col("count_Base"),
            pct_diff=(pl.col("count_Project") - pl.col("count_Base"))
            / pl.col("count_Base"),
        )

        metrics = compute_difference_metrics(df_with_diff)

        gt_table = generate_gt_table(
            df_with_diff,
            configs.get("skims_variable"),
        )

        return mo.accordion(
            {
                f"#### {configs.get('model_name')} (TVD: {round(metrics['tdv'], 4)})": mo.ui.tabs(
                    {
                        "Share": mo.vstack([plot.interactive(), options]),
                        "Table": gt_table,
                    }
                ),
            },
            lazy=True,
        )
    return generate_location_model_ui, get_location_results_options_configs


@app.cell(hide_code=True)
def generate_general_model_diagnostic(
    CONFIGS,
    List,
    Optional,
    alt,
    compute_aggregated_df,
    compute_difference_metrics,
    generate_gt_table,
    pivot_aggregated_df,
    pl,
    scenario_discrete_color_map,
):
    def plot_comparison(agg_df, variable, col, grouping_columns):
        facet_col = grouping_columns[0] if grouping_columns else None

        base = alt.Chart(agg_df).encode(
            alt.X(f"{variable}:N"),
            alt.Y(col)
            .axis(format=".1%" if col == "share" else ".2s")
            .title("Share" if col == "share" else "Count"),
            alt.Color("scenario:N")
            .scale(
                domain=scenario_discrete_color_map.keys(),
                range=scenario_discrete_color_map.values(),
            )
            .title("Scenario")
            .legend(orient="top"),
            xOffset="scenario:N",
            tooltip=col,
        )

        bar = base.mark_bar()

        text = bar.mark_text(
            align="center",
            baseline="middle",
            dy=-8,
            fontSize=CONFIGS.get("dashboard").get("mark_text_fontSize"),
        ).encode(
            alt.Text(f"{col}").format(".1%" if col == "share" else ".2s"),
        )

        chart = bar + text

        if facet_col is not None:
            columns = min(agg_df[facet_col].n_unique(), 3)
            one_facet_width = 900 / columns
            chart = chart.properties(width=one_facet_width).facet(
                facet=f"{facet_col}:N", columns=columns
            )

        return chart.configure_axis(
            labelFontSize=CONFIGS.get("dashboard").get("axis_labelFontSize"),
            titleFontSize=CONFIGS.get("dashboard").get("axis_titleFontSize"),
        )


    # @mo.persistent_cache
    def generate_general_model_diagnostic(
        base: pl.LazyFrame,
        proj: Optional[pl.LazyFrame],
        variable: str,
        by_columns: Optional[List[str]] = None,
        model_name: str = None,
    ):
        """
        Generate diagnostic visuals and a table comparing aggregated Base and Project scenarios.

        Parameters:
            base (pl.LazyFrame): The base scenario data.
            proj (pl.LazyFrame): The project scenario data (can be None).
            variable (str): Primary variable to aggregate by.
            by_columns (Optional[List[str]]): Additional grouping columns.

        Returns:
            A vertically stacked object combining plotly tabs and a formatted table.
        """
        # Determine grouping columns that exist in the base schema
        base_schema = base.collect_schema().keys()
        grouping_columns = (
            [col for col in by_columns if col in base_schema and col != variable]
            if by_columns
            else []
        )
        agg_cols = [variable] + grouping_columns
        agg_cols = set(agg_cols)

        # Compute aggregated data for the Base scenario
        base_agg = compute_aggregated_df(
            base, agg_cols, CONFIGS.get("base").get("weight")
        ).with_columns(scenario=pl.lit("Base"))
        # Compute aggregated data for the Project scenario if available
        if proj is not None:
            proj_agg = (
                compute_aggregated_df(
                    proj, agg_cols, CONFIGS.get("project").get("weight")
                )
                .with_columns(scenario=pl.lit("Project"))
                .cast({variable: base_agg[variable].dtype})
            )
            agg_df = pl.concat([base_agg, proj_agg], how="vertical_relaxed")
        else:
            agg_df = base_agg

        if agg_df[variable].dtype == pl.String:
            agg_df = agg_df.with_columns(
                pl.when(pl.col(variable) == "")
                .then(None)
                .otherwise(pl.col(variable))
            )

        # Calculate the share column within each scenario
        agg_df = agg_df.with_columns(
            share=pl.col("count") / pl.col("count").sum().over("scenario")
        )

        # Pivot the data to compare Base and Project counts side by side
        agg_df_pivoted = pivot_aggregated_df(agg_df, agg_cols)

        metrics = compute_difference_metrics(agg_df_pivoted)

        outs = {
            "Share": plot_comparison(agg_df, variable, "share", grouping_columns),
            "Count": plot_comparison(agg_df, variable, "count", grouping_columns),
            "Table": generate_gt_table(agg_df_pivoted, variable, model_name),
            "metrics": metrics,
        }

        return outs
    return (generate_general_model_diagnostic,)


@app.cell(hide_code=True)
def generate_location_model_diagnostic(
    BASE_OUTPUTS,
    CONFIGS,
    Dict,
    List,
    Optional,
    PROJ_OUTPUTS,
    generate_gt_table,
    mo,
    pivot_aggregated_df,
    pl,
    plotly,
    px,
    scenario_discrete_color_map,
):
    def _generate_scatter(
        lazy_df: pl.LazyFrame,
        scenario_outputs: Dict[str, pl.LazyFrame],
        variable: str,
        land_use_control_variable: str,
        scenario_color: str,
        scenario_name: str,
        by_columns: Optional[List[str]] = None,
    ) -> plotly.graph_objects.Figure:
        """
        Generate a scatter plot comparing aggregated counts to land use control values.

        The function aggregates the given lazy dataframe by the provided variable,
        joins with the land use data to bring in the control variable, and computes
        both the relative and absolute differences. A scatter plot with marginal
        histograms is then created.

        Parameters:
            lazy_df: A Polars LazyFrame containing the base data.
            scenario_outputs: Dictionary with scenario outputs, must include 'land_use'.
            variable: Column name used for grouping.
            land_use_control_variable: Column name from land use data for comparison.
            scenario_color: Color for the scenario's markers.
            by_columns: Optional additional columns for grouping (currently unused).

        Returns:
            A Plotly Express scatter figure.
        """
        # Aggregate data and join with land use control data
        land_use_agg_df = (
            scenario_outputs["land_use"]
            .select("zone_id", land_use_control_variable)
            .join(
                lazy_df.group_by(variable).agg(pl.len().alias("count")),
                right_on=variable,
                left_on="zone_id",
                how="full",
            )
            .drop(variable)
            .rename({"zone_id": variable})
            .with_columns(count=pl.col("count").fill_null(strategy="zero"))
            .with_columns(
                relative_diff=(pl.col("count") / pl.col(land_use_control_variable))
                - 1,
                actual_diff=pl.col("count") - pl.col(land_use_control_variable),
            )
            .sort(variable)
            .collect()
        )

        formatted_land_use_agg_df = land_use_agg_df.with_columns(
            (pl.col("relative_diff") * 100).round(2),
            pl.col(land_use_control_variable, "actual_diff").round(),
        ).rename(
            {
                "relative_diff": "Relative Diff. (%)",
                "actual_diff": "Actual Diff.",
                "count": "Simulated",
                land_use_control_variable: f"Control ({land_use_control_variable})",
            }
        )

        # Create scatter plot with marginal histograms
        fig = px.scatter(
            land_use_agg_df,
            x="relative_diff",
            y="actual_diff",
            marginal_x="histogram",
            marginal_y="histogram",
            hover_data=[variable, "count", land_use_control_variable],
            height=800,
            opacity=0.3,
            title=f"{scenario_name}",
            labels={
                "actual_diff": "Actual difference",
                "relative_diff": "Relative difference",
            },
        )
        fig.update_layout(xaxis=dict(tickformat=".0%"), font=dict(size=16))
        fig.update_traces(marker=dict(color=scenario_color))
        return mo.vstack([fig, formatted_land_use_agg_df])


    def _compute_distance_df(
        lazy_df: pl.LazyFrame,
        scenario_outputs: Dict[str, pl.LazyFrame],
        variable: str,
        skims_variable: str,
        scenario_name: str,
        origin_zone_variable: str,
        by_columns: Optional[List[str]] = None,
        weight_col: str = None,
    ) -> pl.LazyFrame:
        """
        Compute an aggregated distance dataframe for a scenario.

        This helper function filters out rows where the specified variable is 0,
        joins with skims data, groups by the floored skims variable, and counts
        the occurrences. A scenario label is added for later differentiation.

        Parameters:
            lazy_df: A Polars LazyFrame with the input data.
            scenario_outputs: Dictionary with scenario outputs, must include 'skims'.
            variable: Column name to filter and join on (e.g., destination zone).
            skims_variable: Column name in the skims data to group by.
            scenario_name: A label to identify the scenario ("Base" or "Project").

        Returns:
            A Polars LazyFrame with the computed distance aggregations.
        """
        # Determine grouping columns that exist in the base schema
        grouping_columns = by_columns

        query = (
            lazy_df.select(
                grouping_columns + [variable, origin_zone_variable, weight_col]
            )
            .filter(pl.col(variable) > 0)
            .join(
                scenario_outputs["skims"].select(
                    "origin", "destination", skims_variable
                ),
                left_on=[origin_zone_variable, variable],
                right_on=["origin", "destination"],
            )
            .with_columns(pl.col(skims_variable).floor())
            .group_by(grouping_columns + [skims_variable])
        )

        if weight_col is not None:
            query = query.agg(pl.col(weight_col).sum().round().alias("count"))
        else:
            query = query.agg(pl.len().alias("count"))

        return query.with_columns(scenario=pl.lit(scenario_name)).cast(
            {"count": pl.Int64}
        )


    def _generate_distance_plot(
        base_lazy_df: pl.LazyFrame,
        proj_lazy_df: pl.LazyFrame,
        variable: str,
        skims_variable: str,
        origin_zone_variable: str,
        by_columns: Optional[List[str]] = None,
    ) -> plotly.graph_objects.Figure:
        """
        Generate a bar plot comparing distance aggregations between scenarios.

        Two datasets (base and project) are processed to compute distance counts
        based on a skims variable. The results are combined and visualized in a
        grouped bar plot.

        Parameters:
            base_lazy_df: Base scenario data as a Polars LazyFrame.
            proj_lazy_df: Project scenario data as a Polars LazyFrame.
            variable: Column name for joining on zone identifiers.
            skims_variable: Column name in the skims data to group by.
            by_columns: Optional additional columns for grouping (currently unused).

        Returns:
            A Plotly Express bar figure.
        """

        schema = base_lazy_df.collect_schema().names()
        grouping_columns = (
            [col for col in by_columns if col in schema] if by_columns else []
        )

        base_distance = _compute_distance_df(
            base_lazy_df,
            BASE_OUTPUTS,
            variable,
            skims_variable,
            "Base",
            origin_zone_variable,
            grouping_columns,
            CONFIGS.get("base").get("weight"),
        )
        proj_distance = _compute_distance_df(
            proj_lazy_df,
            PROJ_OUTPUTS,
            variable,
            skims_variable,
            "Project",
            origin_zone_variable,
            grouping_columns,
            CONFIGS.get("project").get("weight"),
        )

        # Concatenate and collect the data into a DataFrame
        distance_df = pl.concat(
            [base_distance, proj_distance], how="vertical_relaxed"
        ).collect()

        facet_col = grouping_columns[0] if grouping_columns else None
        facet_order = (
            sorted(distance_df[facet_col].unique())
            if facet_col is not None
            else None
        )

        # Create the bar plot; conversion to pandas may be needed for Plotly Express
        fig = px.bar(
            distance_df,
            x=skims_variable,
            y="count",
            barmode="group",
            color="scenario",
            facet_col=facet_col,
            color_discrete_map=scenario_discrete_color_map,
            hover_data=grouping_columns,
            facet_col_wrap=4,
            height=800,
            text_auto=True,  # ".3s",
            title=f"x-axis: {skims_variable}",
            category_orders={facet_col: facet_order}
            if facet_col is not None
            else None,
        )

        # Remove x-axis titles from all facets
        fig.for_each_xaxis(lambda axis: axis.update(title_text=""))

        fig.update_layout(font=dict(size=16))

        return fig, distance_df


    def generate_location_model_diagnostic(
        base_lazy_df: pl.LazyFrame,
        proj_lazy_df: Optional[pl.LazyFrame],
        variable: str,
        fields: Dict,
        by_columns: Optional[List[str]] = None,
        model_name: str = None,
    ):
        """
        Create a diagnostic UI for the location model with scatter and distance plots.

        This function builds two tabs:
          - 'Differences': Displays scatter plots of relative and actual differences.
          - 'Distance': Shows a bar plot of distance aggregations between scenarios.

        Parameters:
            base_lazy_df: Base scenario data as a Polars LazyFrame.
            proj_lazy_df: Project scenario data as a Polars LazyFrame (optional).
            variable: Column name used for analysis.
            fields: Dictionary containing keys 'skims_variable' and 'land_use_control_variable'.
            by_columns: Optional list of columns for additional grouping (currently unused).

        Returns:
            A UI tabs object combining both diagnostic plots.
        """
        skims_variable = fields.get("skims_variable")
        land_use_control_variable = fields.get("land_use_control_variable")
        origin_zone_variable = fields.get("origin_zone_variable")

        # Build the 'Differences' tab with scatter plots for Base and Project scenarios
        if land_use_control_variable is not None:
            differences_tab = mo.hstack(
                [
                    _generate_scatter(
                        lazy_df=base_lazy_df,
                        scenario_outputs=BASE_OUTPUTS,
                        variable=variable,
                        land_use_control_variable=land_use_control_variable,
                        scenario_color=scenario_discrete_color_map.get("Base"),
                        scenario_name="Base",
                        by_columns=by_columns,
                    ),
                    _generate_scatter(
                        lazy_df=proj_lazy_df,
                        scenario_outputs=PROJ_OUTPUTS,
                        variable=variable,
                        land_use_control_variable=land_use_control_variable,
                        scenario_color=scenario_discrete_color_map.get("Project"),
                        scenario_name="Project",
                        by_columns=by_columns,
                    ),
                ],
                widths="equal",
            )

        # Build the 'Distance' tab with the bar plot
        distance_plot, distance_df = _generate_distance_plot(
            base_lazy_df=base_lazy_df,
            proj_lazy_df=proj_lazy_df,
            variable=variable,
            skims_variable=skims_variable,
            origin_zone_variable=origin_zone_variable,
            by_columns=by_columns,
        )

        # Determine grouping columns that exist in the base schema
        base_schema = base_lazy_df.collect_schema().keys()
        grouping_columns = (
            [col for col in by_columns if col in base_schema and col != variable]
            if by_columns
            else []
        )

        agg_cols = [skims_variable] + grouping_columns
        agg_cols = set(agg_cols)

        pivoted_distance_df = pivot_aggregated_df(distance_df, agg_cols)
        distance_table = generate_gt_table(
            pivoted_distance_df, skims_variable, model_name
        )

        outputs = {
            "data": pivoted_distance_df,
            "distance_df": distance_df,
        }
        return outputs
    return (generate_location_model_diagnostic,)


@app.cell(hide_code=True)
def generate_gt_table(GT, List, cs, loc, md, mo, pl, style, system_fonts):
    def compute_aggregated_df(
        lazy_df: pl.LazyFrame, group_cols: List[str], weight_col: str = None
    ) -> pl.DataFrame:
        """Aggregate the LazyFrame by the provided group columns."""
        query = lazy_df.group_by(group_cols)
        if weight_col is not None:
            query = query.agg(
                count=pl.col(weight_col).sum().round().cast(pl.Int64)
            )
        else:
            query = query.agg(count=pl.len().cast(pl.Int64))
        return query.collect()


    def pivot_aggregated_df(
        agg_df: pl.DataFrame, agg_cols: List[str]
    ) -> pl.DataFrame:
        return (
            agg_df.pivot(
                index=agg_cols,
                on="scenario",
                values=["count"],
                aggregate_function="sum",
            )
            .with_columns(pl.col("Base", "Project").cast(pl.Int64))
            .with_columns(
                share_Base=pl.col("Base") / pl.col("Base").sum(),
                share_Project=pl.col("Project") / pl.col("Project").sum(),
                diff=pl.col("Project") - pl.col("Base"),
            )
            .with_columns(pct_diff=pl.col("diff") / pl.col("Base"))
            .rename({"Base": "count_Base", "Project": "count_Project"})
            .sort(agg_cols)
        )


    def compute_difference_metrics(diff_df):
        metrics = diff_df.select(
            rmse=((pl.col("count_Project") - pl.col("count_Base")) ** 2)
            .mean()
            .sqrt()
            .round(2),
            mape=(
                (
                    (pl.col("count_Project") - pl.col("count_Base")).abs()
                    / pl.col("count_Base")
                ).mean()
                * 100
            ).round(1),
            L1_distance=(pl.col("share_Project") - pl.col("share_Base"))
            .abs()
            .sum(),
        )
        return {
            "rmse": metrics["rmse"].item(),
            "mape": metrics["mape"].item(),
            "tdv": metrics["L1_distance"].item() / 2,
        }


    def generate_gt_table(df: pl.DataFrame, variable: str, table_title: str = ""):
        """Generate a formatted table with RMSE and MAPE metrics."""
        # Compute metrics: RMSE and MAPE
        metrics = compute_difference_metrics(df)

        gt_table = (
            GT(df.sort(variable))
            .tab_header(title=f"Model: {table_title}")
            .tab_spanner(
                label=md("**Share (%)**"), columns=cs.starts_with("share_")
            )
            .tab_spanner(
                label=md("**Count**"),
                columns=[cs.starts_with("count_"), "diff"],
            )
            .tab_style(
                style=[
                    style.fill(color="#edf0ee"),
                ],
                locations=loc.column_header(),
            )
            # Conditional formatting ----------
            # between -0 to -10%
            .tab_style(
                style=style.text(color="#ff917a"),
                locations=loc.body(
                    columns="pct_diff",
                    rows=((pl.col("pct_diff") < 0) & (pl.col("pct_diff") >= -0.1)),
                ),
            )
            # less than - 10%
            .tab_style(
                style=style.text(color="#ff5938"),
                locations=loc.body(
                    columns="pct_diff", rows=pl.col("pct_diff") < -0.1
                ),
            )
            # between 0 to 10%
            .tab_style(
                style=style.text(color="#7aa7ff"),
                locations=loc.body(
                    columns="pct_diff",
                    rows=((pl.col("pct_diff") > 0) & (pl.col("pct_diff") <= 0.1)),
                ),
            )
            # more than 10%
            .tab_style(
                style=style.text(color="#216bff"),
                locations=loc.body(
                    columns="pct_diff", rows=pl.col("pct_diff") > 0.1
                ),
            )
            .cols_move(columns="pct_diff", after="diff")
            .cols_move_to_start(columns=variable)
            .cols_label(
                share_Base=md("**Base**"),
                share_Project=md("**Project**"),
                count_Base=md("**Base**"),
                count_Project=md("**Project**"),
                diff=md("**Difference***"),
                pct_diff=md("% **Difference**"),
            )
            .fmt_percent(columns=[cs.starts_with("share_"), "pct_diff"])
            .fmt_integer(columns=[cs.starts_with("count_"), "diff"])
            .data_color(
                columns=["share_Project", "share_Base"],
                palette="YlGn",
                na_color="lightgray",
            )
            .tab_style(
                style=style.text(weight="bolder"),
                locations=loc.column_header(),
            )
            .tab_source_note(
                source_note=md(
                    f"**Summary Statistics** - RMSE: {metrics['rmse']}, MAPE: {metrics['mape']}%, \\\n Total Variation Distance (TDV): {round(metrics['tdv'], 4)} \\\n *Difference = Project - Base"
                )
            )
            .tab_options(table_font_names=system_fonts("industrial"))
        )

        # Build the formatted table using GT
        return mo.vstack(
            [
                mo.accordion({"Download data": df}),
                gt_table,
            ]
        )
    return (
        compute_aggregated_df,
        compute_difference_metrics,
        generate_gt_table,
        pivot_aggregated_df,
    )


@app.cell(hide_code=True)
def _(mo, results):
    def generate_model_ui(model_name):
        """Generate UI components for a model with TDV metrics display."""

        model_data = results.get(model_name, {})
        if not model_data:
            return mo.accordion(
                {f"#### {model_name} (No data available)": mo.ui.tabs({})}
            )

        # Determine if metrics are at top level or nested
        has_top_level_metrics = "metrics" in model_data

        if has_top_level_metrics:
            # Simple case: metrics at top level
            tdv_value = round(model_data["metrics"].get("tdv", 0), 4)
            results_for_tabs = {
                key: value for key, value in model_data.items() if key != "metrics"
            }
        else:
            # Complex case: metrics nested within each result
            tdv_dict = {
                key: round(value["metrics"]["tdv"], 4)
                for key, value in model_data.items()
                if isinstance(value, dict) and "metrics" in value
            }
            tdv_value = tdv_dict

            results_for_tabs = {}
            for key, result_dict in model_data.items():
                if isinstance(result_dict, dict):
                    filtered_dict = {
                        k: v for k, v in result_dict.items() if k != "metrics"
                    }
                    results_for_tabs[key] = mo.ui.tabs(filtered_dict)

        tabs = mo.ui.tabs(results_for_tabs)
        title = f"#### {model_name} (TDV: {tdv_value})"

        return mo.accordion({title: tabs})
    return (generate_model_ui,)


if __name__ == "__main__":
    app.run()
