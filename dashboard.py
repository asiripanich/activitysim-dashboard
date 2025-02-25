# /// script
# requires-python = ">=3.13"
# dependencies = [
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

__generated_with = "0.11.7"
app = marimo.App(width="full", app_title="ActivitySim dashboard")


@app.cell(hide_code=True)
def import_packages():
    import marimo as mo
    import os
    import polars as pl
    import polars.selectors as cs
    import geopandas as gpd
    from typing import Any, Dict, Optional
    import plotly.express as px
    from great_tables import GT, style, loc, md

    pl.enable_string_cache()
    return Any, Dict, GT, Optional, cs, gpd, loc, md, mo, os, pl, px, style


@app.cell(hide_code=True)
def ui_title(mo):
    mo.hstack(
        [
            # mo.image(
            #     src="https://research.ampo.org/wp-content/uploads/2024/07/activitysim_logo_light.jpg",
            #     height=100,
            # ),
            mo.md("# ActivitySim dashboard"),
        ],
        justify="end",
    )
    return


@app.cell(hide_code=True)
def _():
    scenario_discrete_color_map = {"Base": "#bac5c5", "Project": "#119992"}
    return (scenario_discrete_color_map,)


@app.cell(hide_code=True)
def input_settings(input_dirs_exist, mo, ui_folder_settings_form):
    input_dirs_exist

    base_label = mo.ui.text(
        placeholder="Output folder...", label="**Label:** ", value="Base"
    )

    base_dir = ui_folder_settings_form.value.get("base_dir")

    proj_label = mo.ui.text(
        placeholder="Label...", label="**Label:** ", value="Project"
    )

    proj_dir = ui_folder_settings_form.value.get("proj_dir")

    params_dir = mo.ui.text(
        placeholder="Parameter YAML file..",
        label="**Parameter YAML file** ",
        value=r"example_data/mtc/params.yaml",
        full_width=True,
    )
    return base_dir, base_label, params_dir, proj_dir, proj_label


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
def _(mo, scenario_discrete_color_map):
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
def _(ui_banner):
    ui_banner
    return


@app.cell(hide_code=True)
def ui_folder_settings(mo):
    ui_folder_settings = mo.hstack(
        [mo.md("{base_dir}"), mo.md("{proj_dir}")], widths="equal"
    )
    return (ui_folder_settings,)


@app.cell(hide_code=True)
def ui_folder_settings_form(mo, ui_folder_settings):
    ui_folder_settings_form = ui_folder_settings.batch(
        base_dir=mo.ui.text(
            placeholder="Output folder...",
            label="**Base Folder** ",
            value=r"example_data/mtc/base",
            full_width=True,
        ),
        proj_dir=mo.ui.text(
            placeholder="Output folder...",
            label="**Project Folder** ",
            value=r"example_data/mtc/project",
            full_width=True,
        ),
    ).form()
    return (ui_folder_settings_form,)


@app.cell(hide_code=True)
def ui_folder_settings_form_display(ui_folder_settings_form):
    ui_folder_settings_form
    return


@app.cell
def stop_sign(mo, ui_folder_settings_form):
    mo.stop(
        ui_folder_settings_form.value is None,
        mo.md("**Submit the form to continue.**"),
    )
    check_input_dirs = None
    return (check_input_dirs,)


@app.cell
def check_input_dirs(check_input_dirs, mo, os, ui_folder_settings_form):
    check_input_dirs

    def _check_input_dirs(ui_folder_settings_form_value):
        if not os.path.exists(ui_folder_settings_form_value.get("base_dir")):
            return mo.callout(
                mo.md("## \N{CROSS MARK} **Base folder** was not found."),
                kind="danger",
            )
        if not os.path.exists(ui_folder_settings_form_value.get("proj_dir")):
            return mo.callout(
                mo.md("## \N{CROSS MARK} **Project folder** was not found."),
                kind="danger",
            )
        return True


    _check_result = _check_input_dirs(ui_folder_settings_form.value)
    mo.stop(_check_result is not True, _check_result)

    input_dirs_exist = True
    return (input_dirs_exist,)


@app.cell(hide_code=True)
def read_input_parquets(base_dir, gpd, input_dirs_exist, os, pl, proj_dir):
    input_dirs_exist

    _asim_output_names = {
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


    def read_asim_output(asim_path, attributes):
        filepath = os.path.join(asim_path, attributes["filename"])

        if not os.path.exists(filepath) and attributes["required"] == False:
            return None

        if attributes["filename"] == "zones.parquet":
            return gpd.read_parquet(filepath)
        else:
            return pl.scan_parquet(filepath)


    PROJ_OUTPUTS = {
        table_name: read_asim_output(proj_dir, attributes)
        for table_name, attributes in _asim_output_names.items()
    }
    BASE_OUTPUTS = {
        table_name: read_asim_output(base_dir, attributes)
        for table_name, attributes in _asim_output_names.items()
    }
    return BASE_OUTPUTS, PROJ_OUTPUTS, read_asim_output


@app.cell(hide_code=True)
def ui_overview(input_dirs_exist, mo):
    mo.hstack(
        [mo.md(rf"""# {mo.icon("lucide:chart-bar-big")} Overview""")],
        justify="start",
    ) if input_dirs_exist is True else None
    return


@app.cell(hide_code=True)
def ui_summary_cards(summary_cards):
    summary_cards
    return


@app.cell(hide_code=True)
def _(Any, BASE_OUTPUTS, Dict, MODELS, Optional, PROJ_OUTPUTS, mo, pl):
    def _get_direction(proj_value: float, base_value: float) -> Optional[str]:
        """
        Compare the projected and base values and return a direction string.

        Args:
            proj_value (float): The projected value.
            base_value (float): The base value.

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


    def _compute_summary(outputs):
        """
        Compute summary metrics from output LazyFrames.

        Args:
            outputs (dict): A dictionary containing LazyFrames for keys
                            "persons", "households", "trips", and "tours".

        Returns:
            dict: A dictionary of computed summary metrics.
        """

        def _total_rows(lf: pl.LazyFrame) -> int:
            return lf.select(pl.len()).collect().item()

        def _total_binary_variable(
            lf: pl.LazyFrame, variable: str
        ) -> Optional[int]:
            try:
                result = (
                    lf.group_by(variable)
                    .agg(len=pl.len())
                    .filter(pl.col(variable) == True)
                    .collect()["len"]
                    .item()
                )
            except Exception:
                result = None
            return result

        def _total_categorical_variable(
            lf: pl.LazyFrame, variable: str, category
        ) -> Optional[int]:
            try:
                result = (
                    lf.group_by(variable)
                    .agg(len=pl.len())
                    .filter(pl.col(variable) == category)
                    .collect()["len"]
                    .item()
                )
            except Exception:
                result = None
            return result

        total_persons = outputs["persons"].pipe(_total_rows)
        total_households = outputs["households"].pipe(_total_rows)
        total_trips = outputs["trips"].pipe(_total_rows)
        total_tours = outputs["tours"].pipe(_total_rows)

        return {
            "total_persons": total_persons,
            "total_households": total_households,
            "average_household_size": total_persons / total_households,
            "total_trips": total_trips,
            "total_tours": total_tours,
            "person_trips": total_trips / total_persons,
            "person_tours": total_tours / total_persons,
            "remote_workers": outputs["persons"].pipe(
                _total_binary_variable,
                MODELS["household_person"]
                .get("work_from_home")
                .get("result_field"),
            ),
            "free_parking_at_work": outputs["persons"].pipe(
                _total_binary_variable,
                MODELS["household_person"]
                .get("free_parking_at_work")
                .get("result_field"),
            ),
            "zero-car_households": outputs["households"].pipe(
                _total_categorical_variable,
                MODELS["household_person"]
                .get("auto_ownership")
                .get("result_field"),
                0,
            ),
        }


    def _produce_card(base_value: float, proj_value: float, label: str) -> Any:
        """
        Create a summary card using base and projected values.

        Args:
            base_value (float): The baseline value.
            proj_value (float): The projected value.
            label (str): The label for the card.

        Returns:
            Any: A card object generated by mo.stat.
        """
        formatted_label = label.replace("_", " ").upper()

        if proj_value is None or base_value is None:
            direction = None
            caption = None
        else:
            direction = _get_direction(proj_value, base_value)
            pct_diff = ((proj_value / base_value) - 1) * 100
            decimals = 0 if base_value == int(base_value) else 2
            caption = f"{pct_diff:.1f}% (Base: {base_value:,.{decimals}f})"

        return mo.stat(
            label=formatted_label,
            value=proj_value,
            caption=caption,
            bordered=True,
            direction=direction,
        )


    def _produce_summary_cards(summary_dict: Dict[str, Dict[str, float]]) -> Any:
        """
        Generate summary cards from base and projected value dictionaries.

        Args:
            summary_dict (Dict[str, Dict[str, float]]): A dictionary with keys "base" and "proj",
                each mapping to a dictionary of values.

        Returns:
            Any: A vertically stacked container of summary cards.
        """
        cards = [
            _produce_card(
                summary_dict["base"][key], summary_dict["proj"][key], key
            )
            for key in summary_dict["base"]
        ]
        return mo.vstack(
            [mo.hstack(cards, justify="center", align="center", wrap=True)]
        )


    summary_cards = _produce_summary_cards(
        {
            "base": _compute_summary(BASE_OUTPUTS),
            "proj": _compute_summary(PROJ_OUTPUTS),
        }
    )
    return (summary_cards,)


@app.cell(hide_code=True)
def ui_models(input_dirs_exist, mo):
    mo.hstack(
        [mo.md(rf"""# {mo.icon("lucide:square-chevron-right")} Models""")],
        justify="start",
    ) if input_dirs_exist is True else None
    return


@app.cell(hide_code=True)
def ui_models_helper(column_table, mo):
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
            "#### 🔎 Select a variable from the table to group by": column_table,
        }
    )
    return


@app.cell(hide_code=True)
def column_filter_table(BASE_OUTPUTS, gpd, mo, pl):
    # Define tables to exclude
    _exclude_tables = {"skims", "land_use"}


    # Filter table names and create a list of DataFrames for each valid table
    def _get_columns(frame):
        if isinstance(frame, gpd.GeoDataFrame):
            return frame.columns
        elif isinstance(frame, pl.LazyFrame):
            return frame.collect_schema().keys()


    _dfs = [
        pl.DataFrame(
            {
                "table": table_name,
                "variable": _get_columns(BASE_OUTPUTS[table_name]),
            }
        )
        for table_name in BASE_OUTPUTS.keys()
        if table_name not in _exclude_tables
    ]

    # Concatenate all DataFrames and filter out rows where 'variable' ends with "_id"
    all_columns_df = pl.concat(_dfs).filter(
        ~pl.col("variable").str.ends_with("_id")
    )

    column_table = mo.ui.table(all_columns_df.to_pandas()).form()
    return all_columns_df, column_table


@app.cell(hide_code=True)
def filter_columns(column_table):
    FILTER_COLUMNS = (
        column_table.value["variable"].to_list()
        if column_table.value is not None
        else None
    )
    return (FILTER_COLUMNS,)


@app.cell
def models_settings(pl):
    MODELS = {
        "household_person": {
            "auto_ownership": {
                "table": "households",
                "result_field": "auto_ownership",
            },
            "work_from_home": {
                "table": "persons",
                "result_field": "work_from_home",
            },
            "free_parking_at_work": {
                "table": "persons",
                "result_field": "free_parking_at_work",
            },
            "school_location": {
                "table": "persons",
                "result_field": "school_zone_id",
                "filter_expr": pl.col("school_zone_id") > 0,
                "skims_variable": "DIST",
                "land_use_control_variable": "COLLFTE",
                "origin_zone_variable": "home_zone_id",
            },
            "work_location": {
                "table": "persons",
                "result_field": "workplace_zone_id",
                "filter_expr": pl.col("workplace_zone_id") > 0,
                "skims_variable": "DIST",
                "land_use_control_variable": "TOTEMP",
                "origin_zone_variable": "home_zone_id",
            },
            # 'business_location' is specific to Victoria's implementation 
            "business_location": {
                "table": "persons",
                "result_field": "business_zone_id",
                "filter_expr": pl.col("business_zone_id") > 0,
                "skims_variable": "DIST",
                "land_use_control_variable": "TOTEMP",
                "origin_zone_variable": "home_zone_id",
            },
            "telecommute_frequency": {
                "table": "persons",
                "result_field": "telecommute_frequency",
            },
            "transit_pass_ownership": {
                "table": "persons",
                "result_field": "transit_pass_ownership",
            },
            "transit_pass_subsidy": {
                "table": "persons",
                "result_field": "transit_pass_subsidy",
            },
            "cdap_simulate": {"table": "persons", "result_field": "cdap_activity"},
        },
        "tour": {
            "mandatory_tour_frequency": {
                "table": "persons",
                "result_field": "mandatory_tour_frequency",
                "filter_expr": pl.col("mandatory_tour_frequency") != "",
            },
            "mandatory_tour_scheduling": {
                "table": "tours",
                "result_field": ["start", "end", "duration"],
                "filter_expr": pl.col("tour_category") == "mandatory",
            },
            "joint_tour_composition": {
                "table": "tours",
                "result_field": "composition",
                "filter_expr": pl.col("composition") != "",
            },
            "joint_tour_participation": {
                "table": "tours",
                "result_field": "number_of_participants",
            },
            "joint_tour_scheduling": {
                "table": "tours",
                "result_field": ["start", "end", "duration"],
                "filter_expr": pl.col("tour_category") == "joint",
            },
            "non_mandatory_tour_frequency": {
                "table": "persons",
                "result_field": "non_mandatory_tour_frequency",
                "filter_expr": pl.col("non_mandatory_tour_frequency") != 0,
            },
            "non_mandatory_tour_scheduling": {
                "table": "tours",
                "result_field": ["start", "end", "duration"],
                "filter_expr": pl.col("tour_category") == "non_mandatory",
            },
            "atwork_subtour_frequency": {
                "table": "persons",
                "result_field": "atwork_subtour_frequency",
                "filter_expr": pl.col("atwork_subtour_frequency") != "",
            },
            "atwork_subtour_scheduling": {
                "table": "tours",
                "result_field": ["start", "end", "duration"],
                "filter_expr": pl.col("tour_category") == "atwork",
            },
            "atwork_subtour_mode_choice": {
                "table": "tours",
                "result_field": "tour_mode",
                "filter_expr": pl.col("tour_category") == "atwork",
            },
            "tour_mode_choice_simulate": {
                "table": "tours",
                "result_field": "tour_mode",
            },
            "stop_frequency": {
                "table": "tours",
                "result_field": "stop_frequency",
            },
        },
        "trip": {
            "trip_departure_choice": {
                "table": "trips",
                "result_field": "depart",
            },
            "trip_purpose": {
                "table": "trips",
                "result_field": "purpose",
            },
            "trip_destination": {
                "table": "trips",
                "result_field": "destination",
                "skims_variable": "DIST",
                "origin_zone_variable": "origin",
            },
            "trip_mode": {
                "table": "trips",
                "result_field": "trip_mode",
            },
        },
    }
    return (MODELS,)


@app.cell
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
    def assemble_model_diagnostics(model_name, fields):
        table_name = fields["table"]
        base_lazy_df = BASE_OUTPUTS[table_name]
        proj_lazy_df = PROJ_OUTPUTS[table_name]
        filter_expr = fields.get("filter_expr")

        if filter_expr is not None:
            base_lazy_df = base_lazy_df.filter(filter_expr)
            proj_lazy_df = proj_lazy_df.filter(filter_expr)

        # Ensure result_field is always a list for iteration.
        result_fields = fields["result_field"]
        if not isinstance(result_fields, list):
            result_fields = [result_fields]

        diagnostics = [
            generate_model_diagnostic(
                base_lazy_df,
                proj_lazy_df,
                result_field,
                fields,
                FILTER_COLUMNS,
                model_name,
            )
            for result_field in result_fields
        ]

        # Return a single diagnostic if only one, or stack them otherwise.
        return diagnostics[0] if len(diagnostics) == 1 else mo.vstack(diagnostics)


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
                base_lazy_df, proj_lazy_df, variable, fields, by_columns
            )
        else:
            return generate_general_model_diagnostic(
                base_lazy_df, proj_lazy_df, variable, by_columns
            )


    def check_exists(fields):
        table_cols = BASE_OUTPUTS[fields["table"]].collect_schema().keys()
        result_field = fields["result_field"]

        if isinstance(result_field, str):
            return result_field in table_cols

        return all(item in table_cols for item in result_field)
    return assemble_model_diagnostics, check_exists, generate_model_diagnostic


@app.cell(hide_code=True)
def _(input_dirs_exist, mo):
    mo.md("""### Households/Persons""") if input_dirs_exist is True else None
    return


@app.cell(hide_code=True)
def _(household_person_choices):
    household_person_choices
    return


@app.cell(hide_code=True)
def ui_models_tour_section(input_dirs_exist, mo):
    mo.md("""### Tours""") if input_dirs_exist is True else None
    return


@app.cell(hide_code=True)
def ui_models_tour_choices(tour_choices):
    tour_choices
    return


@app.cell(hide_code=True)
def ui_models_trip_section(input_dirs_exist, mo):
    mo.md("""### Trips""") if input_dirs_exist is True else None
    return


@app.cell(hide_code=True)
def _(trip_choices):
    trip_choices
    return


@app.cell(hide_code=True)
def model_tabs(MODELS, assemble_model_diagnostics, check_exists, mo):
    household_person_choices = mo.accordion(
        {
            f"#### {model_name}": assemble_model_diagnostics(model_name, fields)
            for model_name, fields in MODELS.get("household_person").items()
            if check_exists(fields)
        },
        lazy=True,
    )
    # models_tab_ui = mo.vstack(
    #     [
    #         mo.accordion({"### Households/Persons": household_person_choices}),
    #         mo.accordion({"### Tours": tour_choices}),
    #         mo.accordion({"### Trips": trip_choices}),
    #     ]
    # )
    return (household_person_choices,)


@app.cell(hide_code=True)
def _(MODELS, assemble_model_diagnostics, check_exists, mo):
    tour_choices = mo.accordion(
        {
            f"#### {model_name}": assemble_model_diagnostics(model_name, fields)
            for model_name, fields in MODELS.get("tour").items()
            if check_exists(fields)
        }
    )
    return (tour_choices,)


@app.cell(hide_code=True)
def _(MODELS, assemble_model_diagnostics, check_exists, mo):
    trip_choices = mo.accordion(
        {
            f"#### {model_name}": assemble_model_diagnostics(model_name, fields)
            for model_name, fields in MODELS.get("trip").items()
            if check_exists(fields)
        }
    )
    return (trip_choices,)


@app.cell
def generate_general_model_diagnostic(
    GT,
    List,
    Optional,
    cs,
    loc,
    md,
    mo,
    pl,
    px,
    scenario_discrete_color_map,
    style,
):
    @mo.persistent_cache
    def generate_general_model_diagnostic(
        base: pl.LazyFrame,
        proj: Optional[pl.LazyFrame],
        variable: str,
        by_columns: Optional[List[str]] = None,
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
            [col for col in by_columns if col in base_schema] if by_columns else []
        )
        agg_cols = [variable] + grouping_columns
        agg_cols = set(agg_cols)

        def _compute_aggregate(
            lazy_df: pl.LazyFrame, group_cols: List[str]
        ) -> pl.DataFrame:
            """Aggregate the LazyFrame by the provided group columns."""
            return (
                lazy_df.group_by(group_cols)
                .agg(len=pl.len().cast(pl.Int64))
                .collect()
            )

        # Compute aggregated data for the Base scenario
        base_agg = _compute_aggregate(base, agg_cols).with_columns(
            scenario=pl.lit("Base")
        )
        # Compute aggregated data for the Project scenario if available
        if proj is not None:
            proj_agg = _compute_aggregate(proj, agg_cols).with_columns(
                scenario=pl.lit("Project")
            )
            agg_df = pl.concat([base_agg, proj_agg], how="vertical")
        else:
            agg_df = base_agg

        # Calculate the share column within each scenario
        agg_df = agg_df.with_columns(
            share=pl.col("len") / pl.col("len").sum().over("scenario")
        )

        # Pivot the data to compare Base and Project counts side by side
        agg_df_pivoted = (
            agg_df.pivot(
                index=agg_cols,
                on="scenario",
                values=["len"],
                aggregate_function="sum",
            )
            .with_columns(
                share_Base=pl.col("Base") / pl.col("Base").sum(),
                share_Project=pl.col("Project") / pl.col("Project").sum(),
                diff=pl.col("Project") - pl.col("Base"),
            )
            .with_columns(pct_diff=pl.col("diff") / pl.col("Base"))
            .rename({"Base": "len_Base", "Project": "len_Project"})
            .sort(agg_cols)
        )

        def _generate_formatted_table(df: pl.DataFrame):
            """Generate a formatted table with RMSE and MAPE metrics."""
            # Compute metrics: RMSE and MAPE
            metrics = df.select(
                rmse=((pl.col("len_Project") - pl.col("len_Base")) ** 2)
                .mean()
                .sqrt()
                .round(2),
                mape=(
                    (
                        (pl.col("len_Project") - pl.col("len_Base")).abs()
                        / pl.col("len_Base")
                    ).mean()
                    * 100
                ).round(1),
            )
            rmse = metrics["rmse"].item()
            mape = metrics["mape"].item()

            # Build the formatted table using GT
            return (
                GT(df)
                .tab_header(title=f"RMSE: {rmse}, MAPE: {mape}%")
                .tab_spanner(
                    label=md("**Share (%)**"), columns=cs.starts_with("share_")
                )
                .tab_spanner(
                    label=md("**Count**"), columns=[cs.starts_with("len_"), "diff"]
                )
                .cols_move(columns="pct_diff", after="diff")
                .cols_label(
                    share_Base=md("**Base**"),
                    share_Project=md("**Project**"),
                    len_Base=md("**Base**"),
                    len_Project=md("**Project**"),
                    diff=md("**Project - Base**"),
                    pct_diff=md("% **difference**"),
                )
                .fmt_percent(columns=[cs.starts_with("share_"), "pct_diff"])
                .fmt_integer(columns=[cs.starts_with("len_"), "diff"])
                .data_color(
                    columns=["share_Project", "share_Base"],
                    palette="YlGn",
                    na_color="lightgray",
                )
                .data_color(
                    columns=["pct_diff"],
                    palette="RdBu",
                    domain=[-1.5, 1.5],
                    na_color="lightgray",
                )
                .tab_style(
                    style=style.text(weight="bolder"),
                    locations=loc.column_header(),
                )
            )

        def _generate_figure(col: str):
            """Generate a Plotly bar chart for the specified column ('share' or 'len')."""
            if col == "share":
                labels = {"share": "Percentage (%)"}
                text_auto = ".2%"
            elif col == "len":
                labels = {"len": "Count"}
                text_auto = ".3s"
            else:
                raise ValueError(f"Invalid column specified: {col}")

            fig = px.bar(
                agg_df,
                x=variable,
                y=col,
                color="scenario",
                facet_col=grouping_columns[0] if grouping_columns else None,
                barmode="group",
                color_discrete_map=scenario_discrete_color_map,
                labels=labels,
                text_auto=text_auto,
            )
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.1,
                    xanchor="center",
                    x=0.5,
                ),
                legend_title_text="",
                legend_font_size=14,
            )
            if col == "share":
                fig.update_layout(yaxis=dict(tickformat=".0%"))

            # Remove x-axis titles from all facets
            fig.for_each_xaxis(lambda axis: axis.update(title_text=""))

            # Add a global x-axis label as an annotation
            fig.add_annotation(
                text=variable,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.1,
                showarrow=False,
                font=dict(size=14),
                xanchor="center",
                yanchor="top",
            )
            return fig

        # Generate visuals: create tabs for Share and Count figures and format the table
        tabs = mo.ui.tabs(
            {
                "Share": _generate_figure("share"),
                "Count": _generate_figure("len"),
                "Table": _generate_formatted_table(agg_df_pivoted),
            }
        )

        # Combine visuals and table in a vertical stack layout
        return mo.vstack([tabs])
    return (generate_general_model_diagnostic,)


@app.cell
def _(
    BASE_OUTPUTS,
    Dict,
    List,
    Optional,
    PROJ_OUTPUTS,
    mo,
    pl,
    px,
    scenario_discrete_color_map,
):
    # Assume these globals are defined elsewhere in your project
    # BASE_OUTPUTS, PROJ_OUTPUTS, scenario_discrete_color_map, mo


    def _generate_scatter(
        lazy_df: pl.LazyFrame,
        scenario_outputs: Dict[str, pl.LazyFrame],
        variable: str,
        land_use_control_variable: str,
        scenario_color: str,
        scenario_name: str,
        by_columns: Optional[List[str]] = None,
    ) -> px.Figure:
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
            lazy_df.group_by(variable)
            .agg(pl.len().alias("count"))
            .join(
                scenario_outputs["land_use"].select(
                    "zone_id", land_use_control_variable
                ),
                left_on=variable,
                right_on="zone_id",
                how="full",
            )
            .with_columns(count=pl.col("count"))
            .with_columns(
                relative_diff=(pl.col("count") / pl.col(land_use_control_variable))
                - 1,
                actual_diff=pl.col("count") - pl.col(land_use_control_variable),
            )
            .collect()
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
            title=f"{scenario_name}",
            labels={"actual_diff": "Actual difference", "relative_diff": "Relative difference"}
        )
        fig.update_layout(xaxis=dict(tickformat=".0%"))
        fig.update_traces(marker=dict(color=scenario_color))
        return fig


    def _compute_distance_df(
        lazy_df: pl.LazyFrame,
        scenario_outputs: Dict[str, pl.LazyFrame],
        variable: str,
        skims_variable: str,
        scenario_name: str,
        origin_zone_variable: str,
        by_columns: Optional[List[str]] = None,
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

        return (
            lazy_df.select(grouping_columns + [variable, origin_zone_variable])
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
            .agg(pl.len().alias("count"))
            .with_columns(scenario=pl.lit(scenario_name))
        )


    def _generate_distance_plot(
        base_lazy_df: pl.LazyFrame,
        proj_lazy_df: pl.LazyFrame,
        variable: str,
        skims_variable: str,
        origin_zone_variable: str,
        by_columns: Optional[List[str]] = None,
    ) -> px.Figure:
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
        )
        proj_distance = _compute_distance_df(
            proj_lazy_df,
            PROJ_OUTPUTS,
            variable,
            skims_variable,
            "Project",
            origin_zone_variable,
            grouping_columns,
        )

        # Concatenate and collect the data into a DataFrame
        distance_df = pl.concat([base_distance, proj_distance]).collect()

        # Create the bar plot; conversion to pandas may be needed for Plotly Express
        fig = px.bar(
            distance_df,
            x=skims_variable,
            y="count",
            barmode="group",
            color="scenario",
            facet_col=grouping_columns[0] if grouping_columns else None,
            color_discrete_map=scenario_discrete_color_map,
        )
        return fig


    def generate_location_model_diagnostic(
        base_lazy_df: pl.LazyFrame,
        proj_lazy_df: Optional[pl.LazyFrame],
        variable: str,
        fields: Dict,
        by_columns: Optional[List[str]] = None,
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
        distance_tab = _generate_distance_plot(
            base_lazy_df=base_lazy_df,
            proj_lazy_df=proj_lazy_df,
            variable=variable,
            skims_variable=skims_variable,
            origin_zone_variable=origin_zone_variable,
            by_columns=by_columns,
        )

        # Combine the tabs into a single UI diagnostic object
        if land_use_control_variable is None:
            diagnostic_ui = mo.ui.tabs(
                {
                    "Distance": distance_tab,
                }
            )
        else:
            diagnostic_ui = mo.ui.tabs(
                {
                    "Differences to land use": differences_tab,
                    "Distance": distance_tab,
                }
            )
        return diagnostic_ui
    return (generate_location_model_diagnostic,)


if __name__ == "__main__":
    app.run()
