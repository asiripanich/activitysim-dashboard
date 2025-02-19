# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair==5.5.0",
#     "folium==0.19.4",
#     "geopandas==1.0.1",
#     "great-tables==0.16.1",
#     "leafmap==0.42.9",
#     "mapclassify==2.8.1",
#     "marimo",
#     "matplotlib==3.10.0",
#     "openmatrix==0.3.5.0",
#     "plotly[express]==6.0.0",
#     "polars==1.21.0",
#     "pyarrow==19.0.0",
# ]
# ///

import marimo

__generated_with = "0.11.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import os
    import polars as pl
    import polars.selectors as cs
    import altair as alt
    from typing import Any, Dict, Optional
    import plotly.express as px
    import geopandas as gpd
    from great_tables import GT
    # import openmatrix

    # import leafmap.foliumap as leafmap
    # import folium
    return Any, Dict, GT, Optional, alt, cs, gpd, mo, os, pl, px


@app.cell(hide_code=True)
def _(mo):
    mo.hstack(
        [
            # mo.image(
            #     src="https://research.ampo.org/wp-content/uploads/2024/07/activitysim_logo_light.jpg",
            #     height=100,
            # ),
            mo.md("# ActivitySim dashboard"),
        ],
        justify='end'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    base_label = mo.ui.text(
        placeholder="Output folder...", label="**Label:** ", value="Base"
    )
    base_dir = mo.ui.text(
        placeholder="Output folder...",
        label="**Folder:** ",
        value=r"/Users/amarin/GitHub/asiripanich/activitysim-dashboard/data/output_2018",
        full_width=True,
    )

    proj_label = mo.ui.text(
        placeholder="Label...", label="**Label:** ", value="Project"
    )
    proj_dir = mo.ui.text(
        placeholder="Label...",
        label="**Folder:** ",
        value=r"/Users/amarin/GitHub/asiripanich/activitysim-dashboard/data/output_2026",
        full_width=True,
    )
    return base_dir, base_label, proj_dir, proj_label


@app.cell(hide_code=True)
def _():
    scenario_discrete_color_map = {"Base": "#bac5c5", "Project": "#119992"}
    return (scenario_discrete_color_map,)


@app.cell
def _(mo):
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
              min-width: 500px;
              max-height: 60px;
              align-items: center;
            }
          </style>
        </head>
        """
    )
    return


@app.cell
def _(base_dir, mo, proj_dir, scenario_discrete_color_map):
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(f"""
                        <button class="take-challenge-btn" style='background: linear-gradient(to right, #bac5c5, #ffffff);'>
                            <h1> Base </h1> 
                      </button>"""),
                    # base_label,
                    base_dir,
                ]
            ),
            mo.vstack(
                [
                    mo.md(f"""
                        <button class="take-challenge-btn" style="background: linear-gradient(to left, {scenario_discrete_color_map["Project"]}, #ffffff); color: #ffffff;">
                            <h1> Project </h1>
                      </button>"""),
                    # proj_label,
                    proj_dir,
                ]
            ),
        ],
        justify="space-around",
        align="stretch",
        widths="equal",
    )
    return


@app.cell
def _(
    base_outputs,
    compare_distributions,
    mo,
    proj_outputs,
    summary_cards,
):
    mo.ui.tabs(
        {
            # Overview --------------------------------------------------------------------------------
            f"<h2>{mo.icon('lucide:chart-bar-big')} Overview</h2>": mo.vstack(
                [
                    summary_cards,
                    # mo.accordion(
                    #     {
                    #         f"<h2>{mo.icon('lucide:gauge')} Runtime and memory usage</h2>": mo.hstack(
                    #             []
                    #         )
                    #     },
                    #     lazy=True,
                    # ),
                    # mo.accordion({"<h2>aaa</h2>": mo.hstack([])}, lazy=True)
                    # overview_dualmap,
                ]
            ),
            # Population --------------------------------------------------------------------------------
            f"<h2>{mo.icon('lucide:file-user')} Population</h2>": mo.lazy(
                mo.vstack(
                    [
                        mo.md("## Persons"),
                        mo.hstack(
                            [
                                compare_distributions(
                                    base_outputs, proj_outputs, "persons", "sex"
                                ),
                                # compare_distributions(
                                #     base_outputs, proj_outputs, "persons", "age"
                                # ),
                                compare_distributions(
                                    base_outputs, proj_outputs, "persons", "ptype"
                                ),
                            ],
                            widths="equal",
                        ),
                        mo.md("## Households"),
                        mo.hstack(
                            [
                                compare_distributions(
                                    base_outputs,
                                    proj_outputs,
                                    "households",
                                    "AUTO_OWNERSHIP",
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
                    ]
                ),
                show_loading_indicator=True,
            ),
            # Land use --------------------------------------------------------------------------------
            # f" <h2>{mo.icon('lucide:earth')} Land use</h2>": mo.vstack(
            #     [overview_dualmap]
            # ),
            # trips --------------------------------------------------------------------------------
            f" <h2>{mo.icon('lucide:route')} Trips</h2>": mo.lazy(
                mo.vstack(
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
                ),
                show_loading_indicator=True,
            ),
            # tours --------------------------------------------------------------------------------
            f"<h2>{mo.icon('lucide:repeat')} Tours</h2>": mo.lazy(
                mo.vstack(
                    [
                        mo.hstack(
                            [
                                compare_distributions(
                                    base_outputs,
                                    proj_outputs,
                                    "tours",
                                    "tour_category",
                                ),
                                compare_distributions(
                                    base_outputs,
                                    proj_outputs,
                                    "tours",
                                    "number_of_participants",
                                ),
                            ],
                            widths="equal",
                        ),
                        mo.hstack(
                            [
                                compare_distributions(
                                    base_outputs, proj_outputs, "tours", "start"
                                ),
                                compare_distributions(
                                    base_outputs, proj_outputs, "tours", "end"
                                ),
                            ],
                            widths="equal",
                        ),
                    ]
                ),
                show_loading_indicator=True,
            ),
            # models ------------------------------------------------------------------------------
            # f"<h2>{mo.icon('lucide:square-chevron-right')} Models</h2>": models_tab_ui,
        }
    )
    return


@app.cell
def _(models_tab_ui):
    models_tab_ui
    return


@app.cell
def _(
    base_outputs,
    column_table,
    generate_model_diagnostic,
    mo,
    pl,
    proj_outputs,
    selected_columns,
):
    models_tab_ui = mo.vstack(
        [
            mo.md(f"## {mo.icon('lucide:square-chevron-right')}: Models"),
            mo.accordion(
                {"### Select a variable from the table to group by": column_table}
            ),
            mo.accordion(
                {
                    "### work_from_home": generate_model_diagnostic(
                        base_outputs["persons"],
                        proj_outputs["persons"],
                        "work_from_home",
                        selected_columns,
                    ),
                    "### auto_ownership": generate_model_diagnostic(
                        base_outputs["households"],
                        proj_outputs["households"],
                        "AUTO_OWNERSHIP",
                        selected_columns,
                    ),
                    "<h3>free_parking_at_work</h3>": generate_model_diagnostic(
                        base_outputs["persons"],
                        proj_outputs["persons"],
                        "free_parking_at_work",
                        selected_columns,
                    ),
                    "### telecommute_frequency": generate_model_diagnostic(
                        base_outputs["persons"],
                        proj_outputs["persons"],
                        "telecommute_frequency",
                        selected_columns,
                    ),
                    "### cdap_simulate": generate_model_diagnostic(
                        base_outputs["persons"],
                        proj_outputs["persons"],
                        "cdap_activity",
                        selected_columns,
                    ),
                    "### mandatory_tour_frequency": generate_model_diagnostic(
                        base_outputs["persons"],
                        proj_outputs["persons"],
                        "mandatory_tour_frequency",
                        selected_columns,
                    ),
                    "### mandatory_tour_scheduling": mo.vstack(
                        [
                            generate_model_diagnostic(
                                base_outputs["tours"].filter(
                                    pl.col("tour_category") == "mandatory"
                                ),
                                proj_outputs["tours"].filter(
                                    pl.col("tour_category") == "mandatory"
                                ),
                                col,
                                selected_columns,
                            )
                            for col in ["start", "end", "duration"]
                        ]
                    ),
                    # "### joint_tour_frequency": generate_model_diagnostic(
                    #     base_outputs["persons"],
                    #     proj_outputs["persons"],
                    #     "joint_tour_frequency",
                    #     selected_columns,
                    # ),
                    # "### joint_tour_composition": generate_model_diagnostic(
                    #     base_outputs["persons"],
                    #     proj_outputs["persons"],
                    #     "mandatory_tour_frequency",
                    #     selected_columns,
                    # ),
                    "### non_mandatory_tour_frequency": generate_model_diagnostic(
                        base_outputs["persons"],
                        proj_outputs["persons"],
                        "non_mandatory_tour_frequency",
                        selected_columns,
                    ),
                    "### non_mandatory_tour_scheduling": mo.vstack(
                        [
                            generate_model_diagnostic(
                                base_outputs["tours"].filter(
                                    pl.col("tour_category") == "non_mandatory"
                                ),
                                proj_outputs["tours"].filter(
                                    pl.col("tour_category") == "non_mandatory"
                                ),
                                col,
                                selected_columns,
                            )
                            for col in ["start", "end", "duration"]
                        ]
                    ),
                    "### atwork_subtour_frequency": generate_model_diagnostic(
                        base_outputs["tours"],
                        proj_outputs["tours"],
                        "atwork_subtour_frequency",
                        selected_columns,
                    ),
                    "### tour_mode_choice_simulate": generate_model_diagnostic(
                        base_outputs["tours"],
                        proj_outputs["tours"],
                        "tour_mode",
                        selected_columns,
                    ),
                },
                multiple=True,
                lazy=True,
            ),
        ]
    )
    return (models_tab_ui,)


@app.cell
def _(base_outputs):
    base_outputs['tours'].head().collect()
    return


@app.cell
def column_filter_table(base_outputs, mo, pl):
    all_columns_df = pl.concat(
        [
            pl.DataFrame(
                {"table": table_name, "variable": base_outputs[table_name].columns}
            )
            for table_name in base_outputs.keys()
        ]
    ).filter(~pl.col("variable").str.ends_with("_id"))

    column_table = mo.ui.table(all_columns_df, selection="single")
    return all_columns_df, column_table


@app.cell
def _(column_table):
    selected_columns = (
        column_table.value["variable"].to_list()
        if column_table.value["variable"].len() > 0
        else None
    )
    return (selected_columns,)


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
            value_vars = [col for col in combined.columns if col != "source"]
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
                text_auto=True,
            )

        fig.update_layout(
            yaxis_title="Count",
            xaxis_title=None,
            legend_title_text=None,
            showlegend=False,
        )

        return fig
    return (compare_distributions,)


@app.cell(hide_code=True)
def _(base_dir, os, pl, proj_dir):
    def lazy_read_asim_outputs(asim_path):
        # print(os.path.join(asim_path))

        asim_output_names = {
            "persons": "final_persons.parquet",
            "households": "final_households.parquet",
            "trips": "final_trips.parquet",
            "tours": "final_tours.parquet",
            "joint_tour_participants": "final_joint_tour_participants.parquet",
            "land_use": "final_land_use.parquet",
        }

        # def clean_name(lazy_df):
        #     lazy_df.rename(lambda column_name: column_name.lower())

        lazyframe_dict = {
            key: pl.scan_parquet(os.path.join(asim_path, value))
            for key, value in asim_output_names.items()
        }

        return lazyframe_dict


    # zone_shp = (
    #     gpd.read_file(
    #         r"/Users/amarin/GitHub/tmp/activitysim-prototype-mtc/output/summarize/taz1454.geojson",
    #         use_arrow=True,
    #     )
    #     .to_crs(crs="EPSG:4326")
    #     .rename(columns={"TAZ1454": "zone_id"})
    #     .filter(items=["zone_id", "geometry"])
    # )

    proj_outputs = lazy_read_asim_outputs(proj_dir.value)
    base_outputs = lazy_read_asim_outputs(base_dir.value)

    # base_outputs["land_use"] = zone_shp.merge(
    #     base_outputs["land_use"].collect().to_pandas(), on="zone_id"
    # )
    # proj_outputs["land_use"] = zone_shp.merge(
    #     proj_outputs["land_use"].collect().to_pandas(), on="zone_id"
    # )
    return base_outputs, lazy_read_asim_outputs, proj_outputs


@app.cell
def _(base_outputs, pl, proj_outputs):
    def _compute_summary(outputs):
        def _total_rows(lf):
            return lf.select(pl.len()).collect().item()

        def _total_binary_variable(lf, variable: str):
            return (
                lf.group_by(variable)
                .len()
                .filter(pl.col(variable) == True)
                .collect()["len"]
                .item()
            )

        def _total_catagorical_variable(lf, variable: str, category):
            return (
                lf.group_by(variable)
                .len()
                .filter(pl.col(variable) == category)
                .collect()["len"]
                .item()
            )

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
            "work_from_home": outputs["persons"].pipe(
                _total_binary_variable, "work_from_home"
            ),
            "free_parking_at_work": outputs["persons"].pipe(
                _total_binary_variable, "free_parking_at_work"
            ),
            "free_parking_at_work": outputs["persons"].pipe(
                _total_binary_variable, "free_parking_at_work"
            ),
            "zero-car_households": outputs["households"].pipe(
                _total_catagorical_variable, "AUTO_OWNERSHIP", 0
            ),
            "one-car_households": outputs["households"].pipe(
                _total_catagorical_variable, "AUTO_OWNERSHIP", 1
            ),
            "two-car_households": outputs["households"].pipe(
                _total_catagorical_variable, "AUTO_OWNERSHIP", 2
            ),
            "three_or_more cars_households": outputs["households"].pipe(
                _total_catagorical_variable, "AUTO_OWNERSHIP", 2
            ),
        }


    proj_summary = _compute_summary(proj_outputs)
    base_summary = _compute_summary(base_outputs)
    return base_summary, proj_summary


@app.cell
def _(Any, Dict, Optional, base_summary, mo, proj_summary):
    def get_direction(proj_value: float, base_value: float) -> Optional[str]:
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


    def produce_card(base_value: float, proj_value: float, label: str) -> Any:
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
        direction = get_direction(proj_value, base_value)

        # Calculate the percentage difference.
        pct_diff = ((proj_value / base_value) - 1) * 100

        return mo.stat(
            label=formatted_label,
            value=proj_value,
            caption=f"{pct_diff:.1f}% (Base: {base_value:,.2f})",
            bordered=True,
            direction=direction,
        )


    def produce_summary_cards(summary_dict: Dict[str, Dict[str, float]]) -> Any:
        """
        Generate summary cards from base and projected value dictionaries.

        Args:
            summary_dict (Dict[str, Dict[str, float]]): A dictionary with keys "base" and "proj",
                each mapping to a dictionary of values.

        Returns:
            Any: A vertically stacked container of summary cards.
        """
        cards = [
            produce_card(summary_dict["base"][key], summary_dict["proj"][key], key)
            for key in summary_dict["base"]
        ]

        return mo.vstack(
            [mo.hstack(cards, widths="equal", align="center", wrap=True)]
        )


    # Assuming base_summary and proj_summary are defined elsewhere:
    summary_cards = produce_summary_cards(
        {"base": base_summary, "proj": proj_summary}
    )
    return get_direction, produce_card, produce_summary_cards, summary_cards


@app.cell
def _():
    # def _compute_agg_by(lazy_df: pl.LazyFrame, by_columns: List[str]):
    #     return lazy_df.group_by(by_columns).len().collect()


    # def generate_model_diagnostic(
    #     base: pl.LazyFrame,
    #     proj: pl.LazyFrame,
    #     variable: str,
    #     by_columns: Optional[List[str]] = None,
    # ):
    #     if by_columns is not None:
    #         existed_by_columns = [col for col in by_columns if col in base.columns]
    #     else:
    #         existed_by_columns = []

    #     if len(existed_by_columns) > 0:
    #         agg_cols = [variable] + existed_by_columns
    #     else:
    #         agg_cols = [variable]
    #         existed_by_columns = None

    #     # remove columns that are not in the frames

    #     # Compute aggregated columns
    #     base_agg = _compute_agg_by(base, agg_cols).with_columns(
    #         scenario=pl.lit("Base")
    #     )
    #     if proj is not None:
    #         proj_agg = _compute_agg_by(proj, agg_cols).with_columns(
    #             scenario=pl.lit("Project")
    #         )
    #         agg_df = pl.concat([base_agg, proj_agg], how="vertical")
    #     else:
    #         agg_df = base_agg

    #     # Make plot or further processing here
    #     fig = px.bar(
    #         agg_df,
    #         x=variable,
    #         y="len",
    #         color="scenario",
    #         facet_col=existed_by_columns[0]
    #         if existed_by_columns is not None
    #         else None,
    #         barmode="group",
    #         color_discrete_sequence=px.colors.qualitative.Safe,
    #         labels={"len": "Count"},
    #         text_auto=".4s",
    #     )

    #     return fig


    # generate_model_diagnostic(
    #     base_outputs["persons"],
    #     proj_outputs["persons"],
    #     "work_from_home",
    #     selected_columns,
    # )

    # generate_model_diagnostic(
    #     base_outputs["households"],
    #     proj_outputs["households"],
    #     "AUTO_OWNERSHIP",
    #     None
    # )
    return


@app.cell
def _(GT, List, Optional, cs, mo, pl, px, scenario_discrete_color_map):
    def generate_model_diagnostic(
        base: pl.LazyFrame,
        proj: pl.LazyFrame,
        variable: str,
        by_columns: Optional[List[str]] = None,
    ):
        def _compute_agg_by(lazy_df: pl.LazyFrame, by_columns: List[str]):
            return (
                lazy_df.group_by(by_columns)
                .agg(len=pl.len().cast(pl.Int64))
                .collect()
            )

        def _gen_great_table(tab_df):
            return (
                GT(tab_df)
                .tab_spanner(label="Share (%)", columns=cs.starts_with("share_"))
                .tab_spanner(
                    label="Count", columns=[cs.starts_with("len_"), "diff"]
                )
                .cols_move(columns="pct_diff", after="diff")
                .cols_label(
                    share_Base="Base",
                    share_Project="Project",
                    len_Base="Base",
                    len_Project="Project",
                    diff="Project - Base",
                    pct_diff="% difference",
                )
                .fmt_percent(columns=[cs.starts_with("share_"), "pct_diff"])
                .fmt_integer(columns=[cs.starts_with("len_"), "diff"])
                .data_color(
                    columns=["share_Project", "share_Base"],
                    palette="YlGn",
                    # domain=[0, 1],
                    na_color="lightgray",
                )
                .data_color(
                    columns=["pct_diff"],
                    palette="RdYlBu",
                    domain=[-1, 1],
                    na_color="lightgray",
                )
            )

        def _gen_fig(col):
            if col == "share":
                labels = {"share": "Percentage (%)"}
                text_auto = ".2%"
            elif col == "len":
                labels = {"len": "Count"}
                text_auto = ".3s"
            else:
                return ValueError

            fig = px.bar(
                agg_df,
                x=variable,
                y=col,
                color="scenario",
                facet_col=existed_by_columns[0]
                if existed_by_columns is not None
                else None,
                barmode="group",
                # color_discrete_sequence=px.colors.qualitative.Safe,
                color_discrete_map=scenario_discrete_color_map,
                labels=labels,
                text_auto=text_auto,
            )
            fig.update_layout(
                legend=dict(
                    orientation="h",  # horizontal legend
                    yanchor="bottom",
                    y=1.02,  # just above the top of the plot
                    xanchor="center",
                    x=0.5,
                )
            )

            if col == "share":
                fig.update_layout(
                    yaxis=dict(
                        tickformat=".0%"  # 0 digits after decimal, e.g. 25%
                        # or ".1%" for 25.0%, etc.
                    )
                )

            return fig

        # Check arguments
        if by_columns is not None:
            existed_by_columns = [col for col in by_columns if col in base.columns]
        else:
            existed_by_columns = []

        if len(existed_by_columns) > 0:
            agg_cols = [variable] + existed_by_columns
        else:
            agg_cols = [variable]
            existed_by_columns = None

        # Compute aggregated columns
        base_agg = _compute_agg_by(base, agg_cols).with_columns(
            scenario=pl.lit("Base")
        )
        if proj is not None:
            proj_agg = _compute_agg_by(proj, agg_cols).with_columns(
                scenario=pl.lit("Project")
            )
            agg_df = pl.concat([base_agg, proj_agg], how="vertical")
        else:
            agg_df = base_agg

        agg_df = agg_df.with_columns(
            share=pl.col("len") / pl.col("len").sum().over(pl.col("scenario"))
        )

        # Compute stats
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

        return mo.vstack(
            [mo.ui.tabs(
                    {
                        "Share": _gen_fig("share"),
                        "Count": _gen_fig("len"),
                    }
                ),
                _gen_great_table(agg_df_pivoted)
            ]
            # multiple=True,
            # lazy=True,
        )
    return (generate_model_diagnostic,)


if __name__ == "__main__":
    app.run()
