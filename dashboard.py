# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "folium==0.19.4",
#     "geopandas==1.0.1",
#     "great-tables==0.16.1",
#     "leafmap==0.42.9",
#     "mapclassify==2.8.1",
#     "marimo",
#     "matplotlib==3.10.0",
#     "openmatrix==0.3.5.0",
#     "plotly[express]==6.0.0",
#     "polars==1.22.0",
#     "pyarrow==19.0.0",
#     "pyyaml==6.0.2",
# ]
# ///

import marimo

__generated_with = "0.11.7"
app = marimo.App(width="medium", app_title="ActivitySim dashboard")


@app.cell(hide_code=True)
def import_packages():
    import marimo as mo
    import os
    import polars as pl
    import polars.selectors as cs
    from typing import Any, Dict, Optional
    import plotly.express as px
    import geopandas as gpd
    from great_tables import GT, style, loc, md
    import yaml

    pl.enable_string_cache()
    # import openmatrix

    # import leafmap.foliumap as leafmap
    # import folium
    return (
        Any,
        Dict,
        GT,
        Optional,
        cs,
        gpd,
        loc,
        md,
        mo,
        os,
        pl,
        px,
        style,
        yaml,
    )


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
def input_settings(mo):
    base_label = mo.ui.text(
        placeholder="Output folder...", label="**Label:** ", value="Base"
    )
    base_dir = mo.ui.text(
        placeholder="Output folder...",
        label="**Folder** ",
        value=r"example_data/mtc/base",
        full_width=True,
    )

    proj_label = mo.ui.text(
        placeholder="Label...", label="**Label:** ", value="Project"
    )
    proj_dir = mo.ui.text(
        placeholder="Label...",
        label="**Folder** ",
        value=r"example_data/mtc/project",
        full_width=True,
    )

    params_dir = mo.ui.text(
        placeholder="Parameter YAML file..",
        label="**Parameter YAML file** ",
        value=r"example_data/mtc/params.yaml",
        full_width=True,
    )

    scenario_discrete_color_map = {"Base": "#bac5c5", "Project": "#119992"}
    return (
        base_dir,
        base_label,
        params_dir,
        proj_dir,
        proj_label,
        scenario_discrete_color_map,
    )


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


@app.cell
def _(params_dir, yaml):
    with open(rf"{params_dir.value}", "r") as file:
        PARAMS = yaml.safe_load(file)
    return PARAMS, file


@app.cell(hide_code=True)
def ui_input_settings(base_dir, mo, proj_dir, scenario_discrete_color_map):
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(f"""
                        <button class="take-challenge-btn" style='background: linear-gradient(to right, {scenario_discrete_color_map["Base"]}, #ffffff);'>
                            <h1 style="text-align: left;"> Base </h1> 
                      </button>"""),
                            # base_label,
                            base_dir,
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md(f"""
                        <button class="take-challenge-btn" style="background: linear-gradient(to left, {scenario_discrete_color_map["Project"]}, #ffffff); color: #ffffff;">
                            <h1 style="text-align: right;"> Project </h1>
                      </button>"""),
                            # proj_label,
                            proj_dir,
                        ]
                    ),
                ],
                justify="space-around",
                align="stretch",
                widths="equal",
            ),
            # params_dir,
            mo.md("-------"),
        ]
    )
    return


@app.cell
def read_input_parquets(base_dir, os, pl, proj_dir):
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


    proj_outputs = lazy_read_asim_outputs(proj_dir.value)
    base_outputs = lazy_read_asim_outputs(base_dir.value)

    # zone_shp = (
    #     gpd.read_file(
    #         r"/Users/amarin/GitHub/tmp/activitysim-prototype-mtc/output/summarize/taz1454.geojson",
    #         use_arrow=True,
    #     )
    #     .to_crs(crs="EPSG:4326")
    #     .rename(columns={"TAZ1454": "zone_id"})
    #     .filter(items=["zone_id", "geometry"])
    # )


    # base_outputs["land_use"] = zone_shp.merge(
    #     base_outputs["land_use"].collect().to_pandas(), on="zone_id"
    # )
    # proj_outputs["land_use"] = zone_shp.merge(
    #     proj_outputs["land_use"].collect().to_pandas(), on="zone_id"
    # )
    return base_outputs, lazy_read_asim_outputs, proj_outputs


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


@app.cell
def _(Any, Dict, Optional, PARAMS, base_outputs, mo, pl, proj_outputs):
    def get_model_outcome(key, default_value=None):
        if default_value is None:
            default_value = key
        value = PARAMS.get("models").get(key)
        return value if value is not None else default_value


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
        def _total_rows(lf):
            return lf.select(pl.len()).collect().item()

        def _total_binary_variable(lf, variable: str):
            try:
                _result = (
                    lf.group_by(variable)
                    .len()
                    .filter(pl.col(variable) == True)
                    .collect()["len"]
                    .item()
                )
            except:
                _result = None

            return _result

        def _total_catagorical_variable(lf, variable: str, category):
            try:
                _result = (
                    lf.group_by(variable)
                    .len()
                    .filter(pl.col(variable) == category)
                    .collect()["len"]
                    .item()
                )
            except:
                _result = None

            return _result

        total_persons = outputs["persons"].pipe(_total_rows)
        total_households = outputs["households"].pipe(_total_rows)
        total_trips = outputs["trips"].pipe(_total_rows)
        total_tours = outputs["tours"].pipe(_total_rows)

        return {
            "🧍 total_persons": total_persons,
            "🏠 total_households": total_households,
            "👩🏻‍🤝‍👨🏾 average_household_size": total_persons / total_households,
            "➡️ total_trips": total_trips,
            "🔁 total_tours": total_tours,
            "🧍➡️ person_trips": total_trips / total_persons,
            "🧍🔁 person_tours": total_tours / total_persons,
            "☕️ remote_workers": outputs["persons"].pipe(
                _total_binary_variable, get_model_outcome("work_from_home")
            ),
            "💼🚗 free_parking_at_work": outputs["persons"].pipe(
                _total_binary_variable, get_model_outcome("free_parking_at_work")
            ),
            "0️⃣🚗 zero-car_households": outputs["households"].pipe(
                _total_catagorical_variable, get_model_outcome("auto_ownership"), 0
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
        # Format the label by replacing underscores with spaces and converting to uppercase.
        formatted_label = label.replace("_", " ").upper()

        # Determine the direction based on the comparison of values.
        if proj_value is None or base_value is None:
            direction = None
            pct_diff = None
            caption = None
        else:
            direction = _get_direction(proj_value, base_value)
            pct_diff = ((proj_value / base_value) - 1) * 100
            caption = f"{pct_diff:.1f}% (Base: {base_value:,.{0 if base_value == int(base_value) else 2}f})"

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


    # Assuming base_summary and proj_summary are defined elsewhere:
    summary_cards = _produce_summary_cards(
        {
            "base": _compute_summary(base_outputs),
            "proj": _compute_summary(proj_outputs),
        }
    )
    return get_model_outcome, summary_cards


@app.cell
def _(mo):
    mo.md(r"""--------""")
    return


@app.cell
def ui_models(mo):
    mo.hstack(
        [mo.md(rf"""# {mo.icon("lucide:square-chevron-right")} Models""")],
        justify="start",
    )
    return


@app.cell
def _(column_table, mo):
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
def column_filter_table(base_outputs, mo, pl):
    # Define tables to exclude
    _exclude_tables = {"skims", "land_use"}

    # Filter table names and create a list of DataFrames for each valid table
    _dfs = [
        pl.DataFrame(
            {
                "table": table_name,
                "variable": base_outputs[table_name].collect_schema().keys(),
            }
        )
        for table_name in base_outputs.keys()
        if table_name not in _exclude_tables
    ]

    # Concatenate all DataFrames and filter out rows where 'variable' ends with "_id"
    all_columns_df = pl.concat(_dfs).filter(
        ~pl.col("variable").str.ends_with("_id")
    )

    column_table = mo.ui.table(all_columns_df.to_pandas())
    return all_columns_df, column_table


@app.cell
def _(column_table):
    FILTER_COLUMNS = (
        column_table.value["variable"].to_list()
        if len(column_table.value["variable"]) > 0
        else None
    )
    return (FILTER_COLUMNS,)


@app.cell
def _(pl):
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
            },
            "work_location": {
                "table": "persons",
                "result_field": "workplace_zone_id",
            },
            "business_location": {
                "table": "persons",
                "result_field": "business_zone_id",
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


@app.cell
def _(
    FILTER_COLUMNS,
    base_outputs,
    generate_model_diagnostic,
    mo,
    proj_outputs,
):
    def assemble_model_diagnostics(fields):
        table_name = fields["table"]
        base_lazy_df = base_outputs[table_name]
        proj_lazy_df = proj_outputs[table_name]
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
        table_cols = base_outputs[fields["table"]].collect_schema().keys()
        result_field = fields["result_field"]

        if isinstance(result_field, str):
            return result_field in table_cols

        return all(item in table_cols for item in result_field)
    return assemble_model_diagnostics, check_exists


@app.cell
def model_tabs(MODELS, assemble_model_diagnostics, check_exists, mo):
    household_person_choices = mo.accordion(
        {
            f"#### {key}": assemble_model_diagnostics(fields)
            for key, fields in MODELS.get("household_person").items()
            if check_exists(fields)
        }
    )

    tour_choices = mo.accordion(
        {
            f"#### {key}": assemble_model_diagnostics(fields)
            for key, fields in MODELS.get("tour").items()
            if check_exists(fields)
        }
    )

    trip_choices = mo.accordion(
        {
            f"#### {key}": assemble_model_diagnostics(fields)
            for key, fields in MODELS.get("trip").items()
            if check_exists(fields)
        }
    )


    models_tab_ui = mo.vstack(
        [
            mo.accordion({"### Households/Persons": household_person_choices}),
            mo.accordion({"### Tours": tour_choices}),
            mo.accordion({"### Trips": trip_choices}),
        ]
    )
    return household_person_choices, models_tab_ui, tour_choices, trip_choices


@app.cell
def generate_model_diagnostic(
    GT,
    List,
    Optional,
    cs,
    get_model_outcome,
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
        # Process the outcome variable
        variable = get_model_outcome(variable)

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

            # Build the formatted table using GT (assumed external)
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
            {"Share": generate_figure("share"), "Count": generate_figure("len")}
        )
        formatted_table = generate_formatted_table(agg_df_pivoted)

        # Combine visuals and table in a vertical stack layout
        return mo.vstack([tabs, formatted_table])
    return (generate_model_diagnostic,)


@app.cell
def ui_population(mo):
    mo.hstack(
        [mo.md(rf"""# {mo.icon("lucide:file-user")} Population""")],
        justify="start",
    )
    return


@app.cell
def ui_population_tabs(
    MODELS,
    base_outputs,
    compare_distributions,
    mo,
    proj_outputs,
):
    mo.accordion(
        {
            "### Persons": mo.vstack(
                [
                    mo.hstack(
                        [
                            compare_distributions(
                                base_outputs, proj_outputs, "persons", "sex"
                            ),
                            compare_distributions(
                                base_outputs, proj_outputs, "persons", "ptype"
                            ),
                        ],
                        widths="equal",
                    ),
                    # compare_distributions(
                    #     base_outputs, proj_outputs, "persons", "OCCUPATION"
                    # ),
                ]
            ),
            "### Households": mo.hstack(
                [
                    compare_distributions(
                        base_outputs,
                        proj_outputs,
                        "households",
                        MODELS["household_person"]["auto_ownership"]["result_field"],
                    ),
                    compare_distributions(
                        base_outputs,
                        proj_outputs,
                        "households",
                        "num_workers",
                    ),
                ],
                widths="equal",
            ),
        },
        lazy=True,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""-------""")
    return


@app.cell
def ui_tours(mo):
    mo.hstack([mo.md(rf"""# {mo.icon("lucide:repeat")} Tours""")], justify="start")
    return


@app.cell
def ui_tour_distance_cutoff(cutoff_distance_slider):
    cutoff_distance_slider
    return


@app.cell
def _(mo, plot_dist_to_business, plot_dist_to_school, plot_dist_to_work):
    mo.accordion(
        {
            "### Distance to work": plot_dist_to_work,
            "### Distance to business": plot_dist_to_business,
            "### Distance to school": plot_dist_to_school,
        },
        lazy=True,
        multiple=True,
    )
    return


@app.cell
def _():
    # def compute_agg_by(
    #         lazy_df: pl.LazyFrame, group_cols: List[str]
    #     ) -> pl.DataFrame:
    #         return (
    #             lazy_df.group_by(group_cols)
    #             .agg(len=pl.len().cast(pl.Int64))
    #             .collect()
    #         )

    # def generate_dist_to_mandatory_destinations2(
    #     scenario_outputs,
    #     tour_type: str,
    #     scenario_name: str,
    #     by_columns: List[str] = None,
    # ):

    #     variable = "distance"

    #     # Filter the grouping columns to those that exist in 'base'
    #     existed_by_columns = (
    #         [col for col in by_columns if col in scenario_outputs['tours'].collect_schema().keys()]
    #         if by_columns
    #         else None
    #     )
    #     agg_cols = [variable] + (existed_by_columns if existed_by_columns else [])

    #     # Compute aggregated data for Base and Project scenarios
    #     agg_df = (
    #         scenario_outputs['tours']
    #         .pipe(compute_agg_by, agg_cols)
    #     )

    #     return(agg_df)

    #     # return (
    #     #     scenario_outputs["tours"]
    #     #     .filter(
    #     #         pl.col("tour_category") == "mandatory",
    #     #         pl.col("tour_type") == tour_type,
    #     #     )
    #     #     .join(
    #     #         scenario_outputs["skims"].with_columns(
    #     #             pl.col("origin", "destination").cast(pl.Float64),
    #     #             pl.col("SOV_FREE_DISTANCE__AM").alias("distance"),
    #     #         ),
    #     #         on=["origin", "destination"],
    #     #         how="left",
    #     #     )
    #     #     .with_columns(pl.col("distance").floor())
    #     #     .group_by("distance")
    #     #     .agg(pl.len().alias("count"))
    #     #     .with_columns(
    #     #         share=pl.col("count") / pl.col("count").sum(),
    #     #         scenario=pl.lit(scenario_name),
    #     #     )
    #     #     .collect()
    #     # )

    # _tour_type = "school"

    # _base_dist = generate_dist_to_mandatory_destinations2(
    #     base_outputs, _tour_type, "Base"
    # )
    # _proj_dist = generate_dist_to_mandatory_destinations2(
    #     proj_outputs, _tour_type, "Project"
    # )

    # pl.concat([_base_dist, _proj_dist])
    return


@app.cell
def generate_dist_to_mandatory_destinations(List, PARAMS, pl):
    def generate_dist_to_mandatory_destinations(
        scenario_outputs,
        tour_type: str,
        scenario_name: str,
        by_columns: List[str] = None,
    ):
        return (
            scenario_outputs["tours"]
            .filter(
                pl.col("tour_category") == "mandatory",
                pl.col("tour_type") == tour_type,
            )
            .with_columns(pl.col("origin", "destination").cast(pl.Int64))
            .join(
                scenario_outputs["skims"].with_columns(
                    pl.col("origin", "destination").cast(pl.Int64),
                    pl.col(PARAMS["skims_distance_column"]).alias("distance"),
                ),
                on=["origin", "destination"],
                how="left",
            )
            .with_columns(pl.col("distance").floor())
            .group_by("distance")
            .agg(pl.len().alias("count"))
            .with_columns(
                share=pl.col("count") / pl.col("count").sum(),
                scenario=pl.lit(scenario_name),
            )
            .collect()
        )
    return (generate_dist_to_mandatory_destinations,)


@app.cell
def _(dist_school_tours, dist_work_tours, mo):
    _max_distance = max(
        [
            dist_work_tours["distance"].max(),
            # dist_business_tours["distance"].max(),
            dist_school_tours["distance"].max(),
        ]
    )

    cutoff_distance_slider = mo.ui.slider(
        start=0,
        stop=_max_distance,
        label="Cut-off distance (KM)",
        value=min(_max_distance, 80),
    )
    return (cutoff_distance_slider,)


@app.cell
def _(base_outputs, generate_dist_to_mandatory_destinations, pl, proj_outputs):
    _tour_type = "school"

    _base_dist = generate_dist_to_mandatory_destinations(
        base_outputs, _tour_type, "Base"
    )
    _proj_dist = generate_dist_to_mandatory_destinations(
        proj_outputs, _tour_type, "Project"
    )

    dist_school_tours = pl.concat([_base_dist, _proj_dist])
    return (dist_school_tours,)


@app.cell
def _(base_outputs, generate_dist_to_mandatory_destinations, pl, proj_outputs):
    _tour_type = "business"

    _base_dist = generate_dist_to_mandatory_destinations(
        base_outputs, _tour_type, "Base"
    )
    _proj_dist = generate_dist_to_mandatory_destinations(
        proj_outputs, _tour_type, "Project"
    )

    dist_business_tours = pl.concat([_base_dist, _proj_dist])
    return (dist_business_tours,)


@app.cell
def _(base_outputs, generate_dist_to_mandatory_destinations, pl, proj_outputs):
    _tour_type = "work"

    _base_dist = generate_dist_to_mandatory_destinations(
        base_outputs, _tour_type, "Base"
    )
    _proj_dist = generate_dist_to_mandatory_destinations(
        proj_outputs, _tour_type, "Project"
    )

    dist_work_tours = pl.concat([_base_dist, _proj_dist])
    return (dist_work_tours,)


@app.cell
def _(pl, px, scenario_discrete_color_map):
    def plot_dist_to_mand_tour(_dist_tours, cutoff_distance_slider):
        _fig = px.bar(
            _dist_tours.filter(pl.col("distance") <= cutoff_distance_slider.value),
            y="share",
            x="distance",
            color="scenario",
            barmode="group",
            subtitle=f"Cut-off distance at {cutoff_distance_slider.value}KM",
            labels={"share": "Proportion (%)", "distance": "Distance"},
            color_discrete_map=scenario_discrete_color_map,
        )

        _fig.update_layout(yaxis=dict(tickformat=".0%"))

        return _fig
    return (plot_dist_to_mand_tour,)


@app.cell
def _(cutoff_distance_slider, dist_work_tours, plot_dist_to_mand_tour):
    plot_dist_to_work = plot_dist_to_mand_tour(
        dist_work_tours, cutoff_distance_slider
    )
    return (plot_dist_to_work,)


@app.cell
def _(cutoff_distance_slider, dist_business_tours, plot_dist_to_mand_tour):
    plot_dist_to_business = plot_dist_to_mand_tour(
        dist_business_tours, cutoff_distance_slider
    )
    return (plot_dist_to_business,)


@app.cell
def _(cutoff_distance_slider, dist_school_tours, plot_dist_to_mand_tour):
    plot_dist_to_school = plot_dist_to_mand_tour(
        dist_school_tours, cutoff_distance_slider
    )
    return (plot_dist_to_school,)


@app.cell
def _(mo):
    mo.md(r"""-------""")
    return


@app.cell
def _(mo):
    mo.hstack([mo.md(rf"""# {mo.icon("lucide:route")} Trips""")], justify="start")
    return


@app.cell
def _(base_outputs, compare_distributions, mo, proj_outputs):
    mo.accordion(
        {
            "## Plots": mo.vstack(
                [
                    mo.hstack(
                        [
                            compare_distributions(
                                base_outputs,
                                proj_outputs,
                                "trips",
                                "trip_mode",
                            ),
                            compare_distributions(
                                base_outputs, proj_outputs, "trips", "depart"
                            ),
                            compare_distributions(
                                base_outputs,
                                proj_outputs,
                                "trips",
                                "primary_purpose",
                            ),
                        ],
                        widths="equal",
                    ),
                ]
            )
        },
        lazy=True,
    )
    return


@app.cell
def ui_profiling(mo):
    mo.vstack(
        [
            mo.hstack(
                [mo.md(rf"""# {mo.icon("lucide:gauge")} Profiling""")],
                justify="start",
            ),
            mo.accordion(
                {
                    "Work in progress": mo.callout(
                        """
            This section will compare the model runtime and memory usage of the two scenarios.
            """,
                        kind="warn",
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(folium):
    def create_dualmap_leaflet(gdf):
        # find centre of the shp file
        center_y = gdf.geometry.centroid.y.mean()
        center_x = gdf.geometry.centroid.x.mean()

        # create a dualmap object
        m = folium.plugins.DualMap(
            location=(center_y, center_x), tiles=None, zoom_start=13
        )

        # create basemaps
        folium.TileLayer("cartodbpositron").add_to(m.m1)
        folium.TileLayer("cartodbpositron").add_to(m.m2)

        # add polygons
        gdf[["zone_id", "employment_density", "geometry"]].explore(
            m=m.m1, column="employment_density"
        )
        gdf[["zone_id", "household_density", "geometry"]].explore(
            m=m.m2, column="household_density"
        )

        # add options
        folium.LayerControl(collapsed=False).add_to(m)

        return m


    # overview_dualmap = create_dualmap_leaflet(base_outputs["land_use"])
    return (create_dualmap_leaflet,)


@app.cell(hide_code=True)
def _(Dict, List, Optional, Union, pl, px, scenario_discrete_color_map):
    def compare_distributions(
        base: Dict,
        proj: Dict,
        table: str,
        variable: Optional[Union[str, List[str]]] = None,
    ) -> px.fig:
        """
        Compare the distributions of specified variable(s) in two Lazy Polars DataFrames using Altair histograms.

        The function collects the LazyFrames, adds a 'source' column to distinguish the datasets,
        and reshapes the combined data into a long format without converting to Pandas.
        The resulting data is passed to Altair as a list of dictionaries.
        """

        # Add a 'source' column to each DataFrame.
        df1 = base[table].with_columns(pl.lit("Base").alias("source"))
        df2 = proj[table].with_columns(pl.lit("Project").alias("source"))

        # Combine the two DataFrames.
        combined = pl.concat([df1, df2])

        # Determine which columns to include. Exclude the 'source' column.
        if variable is None:
            value_vars = [
                col for col in combined.collect_schema().columns if col != "source"
            ]
        elif isinstance(variable, str):
            value_vars = [variable]
        else:
            value_vars = variable

        unique_count = (
            combined.select(pl.col(variable).n_unique()).collect().item()
        )

        plot_title = variable.upper()

        if unique_count > 50:
            fig = px.histogram(
                combined.select(pl.col("source", variable)).collect(),
                x=variable,
                color="source",
                barmode="overlay",
                title=f"{plot_title} Histogram",
                nbins=50,  # Adjust the number of bins as needed.
            )
        else:
            agg_data = (
                combined.group_by("source", variable)
                .len()
                .with_columns(
                    pl.col(variable).cast(pl.String).alias("_variable_string")
                )
                .sort("_variable_string")
                .collect()
            )
            fig = px.bar(
                agg_data,
                x=variable,
                y="len",
                color="source",
                barmode="group",
                title=plot_title,
                color_discrete_map=scenario_discrete_color_map,
                text_auto=".3s",
            )

        fig.update_layout(
            yaxis_title="Count",
            xaxis_title=None,
            legend_title_text=None,
            showlegend=False,
        )

        return fig
    return (compare_distributions,)


if __name__ == "__main__":
    app.run()
