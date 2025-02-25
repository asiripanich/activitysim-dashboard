# /// script
# requires-python = ">=3.13"
# dependencies = [
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
    from typing import Any, Dict, Optional
    import plotly.express as px
    from great_tables import GT, style, loc, md
    return Any, Dict, GT, Optional, cs, loc, md, mo, os, pl, px, style


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
              min-width: 535px;
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
        justify="space-around",
        align="stretch",
        widths="equal",
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
def _(mo, ui_folder_settings_form):
    mo.stop(ui_folder_settings_form.value is None, mo.md("**Submit the form to continue.**"))
    check_input_dirs = None
    return (check_input_dirs,)


@app.cell
def check_input_dirs(check_input_dirs, mo, os, ui_folder_settings_form):
    check_input_dirs
    def _check_input_dirs(ui_folder_settings_form_value):
        if not os.path.exists(ui_folder_settings_form_value.get("base_dir")):
            return mo.callout(mo.md("## \N{CROSS MARK} **Base folder** was not found."), kind="danger")
        if not os.path.exists(ui_folder_settings_form_value.get("proj_dir")):
            return mo.callout(mo.md("## \N{CROSS MARK} **Project folder** was not found."), kind="danger")
        return True

    _check_result = _check_input_dirs(ui_folder_settings_form.value)
    mo.stop(_check_result is not True, _check_result)

    input_dirs_exist = True
    return (input_dirs_exist,)

    _asim_output_names = {
        "persons": "final_persons.parquet",
        "households": "final_households.parquet",
        "trips": "final_trips.parquet",
        "tours": "final_tours.parquet",
        "joint_tour_participants": "final_joint_tour_participants.parquet",
        "land_use": "final_land_use.parquet",
        "skims": "skims.parquet",
    }


    def lazy_read_asim_outputs(asim_path):
        return {
            key: pl.scan_parquet(os.path.join(asim_path, value))
            for key, value in _asim_output_names.items()
        }


    PROJ_OUTPUTS = lazy_read_asim_outputs(proj_dir.value)
    BASE_OUTPUTS = lazy_read_asim_outputs(base_dir.value)
    return BASE_OUTPUTS, PROJ_OUTPUTS, lazy_read_asim_outputs


@app.cell
def ui_overview(mo):
    mo.hstack(
        [mo.md(rf"""# {mo.icon("lucide:chart-bar-big")} Overview""")],
        justify="start",
    )
    return


@app.cell
def _(summary_cards):
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
def ui_models(mo):
    mo.hstack(
        [mo.md(rf"""# {mo.icon("lucide:square-chevron-right")} Models""")],
        justify="start",
    )
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
def ui_model_tabs(models_tab_ui):
    models_tab_ui
    return


@app.cell(hide_code=True)
def column_filter_table(BASE_OUTPUTS, mo, pl):
    # Define tables to exclude
    _exclude_tables = {"skims", "land_use"}

    # Filter table names and create a list of DataFrames for each valid table
    _dfs = [
        pl.DataFrame(
            {
                "table": table_name,
                "variable": BASE_OUTPUTS[table_name].collect_schema().keys(),
            }
        )
        for table_name in BASE_OUTPUTS.keys()
        if table_name not in _exclude_tables
    ]

    # Concatenate all DataFrames and filter out rows where 'variable' ends with "_id"
    all_columns_df = pl.concat(_dfs).filter(
        ~pl.col("variable").str.ends_with("_id")
    )

    column_table = mo.ui.table(all_columns_df.to_pandas())
    return all_columns_df, column_table


@app.cell(hide_code=True)
def filter_columns(column_table):
    FILTER_COLUMNS = (
        column_table.value["variable"].to_list()
        if len(column_table.value["variable"]) > 0
        else None
    )
    return (FILTER_COLUMNS,)


@app.cell(hide_code=True)
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
            },
            "work_location": {
                "table": "persons",
                "result_field": "workplace_zone_id",
                "filter_expr": pl.col("workplace_zone_id") > 0,
            },
            "business_location": {
                "table": "persons",
                "result_field": "business_zone_id",
                "filter_expr": pl.col("business_zone_id") > 0,
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
            },
            "trip_mode": {
                "table": "trips",
                "result_field": "trip_mode",
            },
        },
    }
    return (MODELS,)


@app.cell(hide_code=True)
def assemble_model_diagnostics(
    BASE_OUTPUTS,
    FILTER_COLUMNS,
    PROJ_OUTPUTS,
    generate_model_diagnostic,
    mo,
):
    def assemble_model_diagnostics(fields):
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
                field,
                FILTER_COLUMNS,
            )
            for field in result_fields
        ]

        # Return a single diagnostic if only one, or stack them otherwise.
        return diagnostics[0] if len(diagnostics) == 1 else mo.vstack(diagnostics)


    def check_exists(fields):
        table_cols = BASE_OUTPUTS[fields["table"]].collect_schema().keys()
        result_field = fields["result_field"]

        if isinstance(result_field, str):
            return result_field in table_cols

        return all(item in table_cols for item in result_field)
    return assemble_model_diagnostics, check_exists


@app.cell
def _(input_dirs_exist, mo):
    mo.md("""### Households/Persons""") if input_dirs_exist is True else None
    return


@app.cell
def _(household_person_choices):
    household_person_choices
    return


@app.cell
def _(input_dirs_exist, mo):
    mo.md("""### Tours""") if input_dirs_exist is True else None
    return


@app.cell
def _(tour_choices):
    tour_choices
    return


@app.cell
def _(input_dirs_exist, mo):
    mo.md("""### Trips""") if input_dirs_exist is True else None
    return


@app.cell
def _(trip_choices):
    trip_choices
    return


@app.cell(hide_code=True)
def model_tabs(MODELS, assemble_model_diagnostics, check_exists, mo):
    household_person_choices = mo.accordion(
        {
            f"#### {key}": assemble_model_diagnostics(fields)
            for key, fields in MODELS.get("household_person").items()
            if check_exists(fields)
        }, lazy=True
    )
    # models_tab_ui = mo.vstack(
    #     [
    #         mo.accordion({"### Households/Persons": household_person_choices}),
    #         mo.accordion({"### Tours": tour_choices}),
    #         mo.accordion({"### Trips": trip_choices}),
    #     ]
    # )
    return (household_person_choices,)


@app.cell
def _(MODELS, assemble_model_diagnostics, check_exists, mo):
    tour_choices = mo.accordion(
        {
            f"#### {key}": assemble_model_diagnostics(fields)
            for key, fields in MODELS.get("tour").items()
            if check_exists(fields)
        }
    )
    return (tour_choices,)


@app.cell
def _(MODELS, assemble_model_diagnostics, check_exists, mo):
    trip_choices = mo.accordion(
        {
            f"#### {key}": assemble_model_diagnostics(fields)
            for key, fields in MODELS.get("trip").items()
            if check_exists(fields)
        }
    )
    return (trip_choices,)


@app.cell(hide_code=True)
def generate_model_diagnostic(
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
    def generate_model_diagnostic(
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

        def compute_aggregate(
            lazy_df: pl.LazyFrame, group_cols: List[str]
        ) -> pl.DataFrame:
            """Aggregate the LazyFrame by the provided group columns."""
            return (
                lazy_df.group_by(group_cols)
                .agg(len=pl.len().cast(pl.Int64))
                .collect()
            )

        # Compute aggregated data for the Base scenario
        base_agg = compute_aggregate(base, agg_cols).with_columns(
            scenario=pl.lit("Base")
        )
        # Compute aggregated data for the Project scenario if available
        if proj is not None:
            proj_agg = compute_aggregate(proj, agg_cols).with_columns(
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

        def generate_formatted_table(df: pl.DataFrame):
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

        def generate_figure(col: str):
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
                "Share": generate_figure("share"),
                "Count": generate_figure("len"),
                "Table": generate_formatted_table(agg_df_pivoted),
            }
        )

        # Combine visuals and table in a vertical stack layout
        return mo.vstack([tabs])
    return (generate_model_diagnostic,)


if __name__ == "__main__":
    app.run()
