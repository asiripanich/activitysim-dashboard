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
    from typing import Any, Dict, Optional
    import plotly.express as px
    import geopandas as gpd
    from great_tables import GT

    # import openmatrix

    # import leafmap.foliumap as leafmap
    # import folium
    return Any, Dict, GT, Optional, cs, gpd, mo, os, pl, px


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
        justify="end",
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

    scenario_discrete_color_map = {"Base": "#bac5c5", "Project": "#119992"}
    return (
        base_dir,
        base_label,
        proj_dir,
        proj_label,
        scenario_discrete_color_map,
    )


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _():
    # mo.ui.tabs(
    #     {
    #         # Overview --------------------------------------------------------------------------------
    #         f"<h2>{mo.icon('lucide:chart-bar-big')} Overview</h2>": mo.vstack(
    #             [
    #                 summary_cards,
    #                 # mo.accordion(
    #                 #     {
    #                 #         f"<h2>{mo.icon('lucide:gauge')} Runtime and memory usage</h2>": mo.hstack(
    #                 #             []
    #                 #         )
    #                 #     },
    #                 #     lazy=True,
    #                 # ),
    #                 # mo.accordion({"<h2>aaa</h2>": mo.hstack([])}, lazy=True)
    #                 # overview_dualmap,
    #             ]
    #         ),
    #         # Models -----------------
    #         # f"<h2>{mo.icon('lucide:square-chevron-right')} Models</h2>": mo.vstack([models_tab_ui])
    #         # ,
    #         # Population --------------------------------------------------------------------------------
    #         f"<h2>{mo.icon('lucide:file-user')} Population</h2>": mo.lazy(
    #             mo.vstack(
    #                 [
    #                     mo.md("## Persons"),
    #                     mo.hstack(
    #                         [
    #                             compare_distributions(
    #                                 base_outputs, proj_outputs, "persons", "sex"
    #                             ),
    #                             # compare_distributions(
    #                             #     base_outputs, proj_outputs, "persons", "age"
    #                             # ),
    #                             compare_distributions(
    #                                 base_outputs, proj_outputs, "persons", "ptype"
    #                             ),
    #                         ],
    #                         widths="equal",
    #                     ),
    #                     mo.md("## Households"),
    #                     mo.hstack(
    #                         [
    #                             compare_distributions(
    #                                 base_outputs,
    #                                 proj_outputs,
    #                                 "households",
    #                                 "AUTO_OWNERSHIP",
    #                             ),
    #                             compare_distributions(
    #                                 base_outputs,
    #                                 proj_outputs,
    #                                 "households",
    #                                 "num_workers",
    #                             ),
    #                         ],
    #                         widths="equal",
    #                     ),
    #                 ]
    #             ),
    #             show_loading_indicator=True,
    #         ),
    #         # Land use --------------------------------------------------------------------------------
    #         # f" <h2>{mo.icon('lucide:earth')} Land use</h2>": mo.vstack(
    #         #     [overview_dualmap]
    #         # ),
    #         # trips --------------------------------------------------------------------------------
    #         f" <h2>{mo.icon('lucide:route')} Trips</h2>": mo.lazy(
    #             mo.vstack(
    #                 [
    #                     #         # mo.hstack(
    #                     #         #     [
    #                     #         #         compare_distributions(
    #                     #         #             base_outputs,
    #                     #         #             proj_outputs,
    #                     #         #             "trips",
    #                     #         #             "trip_mode",
    #                     #         #         ),
    #                     #         #         compare_distributions(
    #                     #         #             base_outputs, proj_outputs, "trips", "depart"
    #                     #         #         ),
    #                     #         #         compare_distributions(
    #                     #         #             base_outputs,
    #                     #         #             proj_outputs,
    #                     #         #             "trips",
    #                     #         #             "primary_purpose",
    #                     #         #         ),
    #                     #         #     ],
    #                     #         #     widths="equal",
    #                     #         # ),
    #                 ]
    #             ),
    #             show_loading_indicator=True,
    #         ),
    #         # tours --------------------------------------------------------------------------------
    #         f"<h2>{mo.icon('lucide:repeat')} Tours</h2>": mo.lazy(
    #             mo.vstack(
    #                 [
    #                     cutoff_distance_slider,
    #                     plot_dist_to_mandatory_destinations,
    #                     # mo.hstack(
    #                     #     [
    #                     #         compare_distributions(
    #                     #             base_outputs,
    #                     #             proj_outputs,
    #                     #             "tours",
    #                     #             "tour_category",
    #                     #         ),
    #                     #         compare_distributions(
    #                     #             base_outputs,
    #                     #             proj_outputs,
    #                     #             "tours",
    #                     #             "number_of_participants",
    #                     #         ),
    #                     #     ],
    #                     #     widths="equal",
    #                     # ),
    #                     # mo.hstack(
    #                     #     [
    #                     #         compare_distributions(
    #                     #             base_outputs, proj_outputs, "tours", "start"
    #                     #         ),
    #                     #         compare_distributions(
    #                     #             base_outputs, proj_outputs, "tours", "end"
    #                     #         ),
    #                     #     ],
    #                     #     widths="equal",
    #                     # ),
    #                 ]
    #             ),
    #             show_loading_indicator=True,
    #         ),
    #         # models ------------------------------------------------------------------------------
    #         # f"<h2>{mo.icon('lucide:square-chevron-right')} Models</h2>": models_tab_ui,
    #     }
    # )
    return


@app.cell
def _(mo):
    mo.md(r"""-------""")
    return


@app.cell
def _(mo):
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
def _(mo):
    mo.md(r"""--------""")
    return


@app.cell
def _(mo):
    mo.hstack(
        [mo.md(rf"""# {mo.icon("lucide:square-chevron-right")} Models""")],
        justify="start",
    )
    return


@app.cell
def _(column_table, mo):
    mo.accordion(
        {"### Select a variable from the table to group by": column_table}
    )
    return


@app.cell
def _(models_tab_ui):
    models_tab_ui
    return


@app.cell
def _(
    base_outputs,
    generate_model_diagnostic,
    mo,
    pl,
    proj_outputs,
    selected_columns,
):
    long_term_choices = mo.accordion(
        {
            "#### work_from_home": generate_model_diagnostic(
                base_outputs["persons"],
                proj_outputs["persons"],
                "work_from_home",
                selected_columns,
            ),
            "#### auto_ownership": generate_model_diagnostic(
                base_outputs["households"],
                proj_outputs["households"],
                "AUTO_OWNERSHIP",
                selected_columns,
            ),
            "#### free_parking_at_work": generate_model_diagnostic(
                base_outputs["persons"],
                proj_outputs["persons"],
                "free_parking_at_work",
                selected_columns,
            ),
            "#### telecommute_frequency": generate_model_diagnostic(
                base_outputs["persons"].filter(
                    pl.col("telecommute_frequency") != ""
                ),
                proj_outputs["persons"].filter(
                    pl.col("telecommute_frequency") != ""
                ),
                "telecommute_frequency",
                selected_columns,
            ),
            "#### cdap_simulate": generate_model_diagnostic(
                base_outputs["persons"],
                proj_outputs["persons"],
                "cdap_activity",
                selected_columns,
            ),
        },
        # lazy=True,
    )

    tour_choices = mo.accordion(
        {
            "#### mandatory_tour_frequency": generate_model_diagnostic(
                base_outputs["persons"],
                proj_outputs["persons"],
                "mandatory_tour_frequency",
                selected_columns,
            ),
            "#### mandatory_tour_scheduling": mo.vstack(
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
            # "#### joint_tour_frequency": generate_model_diagnostic(
            #     base_outputs["persons"],
            #     proj_outputs["persons"],
            #     "joint_tour_frequency",
            #     selected_columns,
            # ),
            "#### joint_tour_composition": generate_model_diagnostic(
                base_outputs["tours"].filter(pl.col("composition") != ""),
                proj_outputs["tours"].filter(pl.col("composition") != ""),
                "composition",
                selected_columns,
            ),
            "#### joint_tour_participation": generate_model_diagnostic(
                base_outputs["tours"],
                proj_outputs["tours"],
                "number_of_participants",
                selected_columns,
            ),
            "#### joint_tour_scheduling": mo.vstack(
                [
                    generate_model_diagnostic(
                        base_outputs["tours"].filter(
                            pl.col("tour_category") == "joint"
                        ),
                        proj_outputs["tours"].filter(
                            pl.col("tour_category") == "joint"
                        ),
                        col,
                        selected_columns,
                    )
                    for col in ["start", "end", "duration"]
                ]
            ),
            "#### non_mandatory_tour_frequency": generate_model_diagnostic(
                base_outputs["persons"],
                proj_outputs["persons"],
                "non_mandatory_tour_frequency",
                selected_columns,
            ),
            "#### non_mandatory_tour_scheduling": mo.vstack(
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
            "#### atwork_subtour_frequency": generate_model_diagnostic(
                base_outputs["tours"].filter(
                    ~pl.col("atwork_subtour_frequency").is_in([""])
                ),
                proj_outputs["tours"].filter(
                    ~pl.col("atwork_subtour_frequency").is_in([""])
                ),
                "atwork_subtour_frequency",
                selected_columns,
            ),
            "#### atwork_subtour_scheduling": mo.vstack(
                [
                    generate_model_diagnostic(
                        base_outputs["tours"].filter(
                            pl.col("tour_category") == "atwork"
                        ),
                        proj_outputs["tours"].filter(
                            pl.col("tour_category") == "atwork"
                        ),
                        col,
                        selected_columns,
                    )
                    for col in ["start", "end", "duration"]
                ]
            ),
            "#### atwork_subtour_mode_choice": generate_model_diagnostic(
                base_outputs["tours"].filter(pl.col("tour_category") == "atwork"),
                proj_outputs["tours"].filter(pl.col("tour_category") == "atwork"),
                "tour_mode",
                selected_columns,
            ),
            "#### tour_mode_choice_simulate": generate_model_diagnostic(
                base_outputs["tours"],
                proj_outputs["tours"],
                "tour_mode",
                selected_columns,
            ),
            "#### stop_frequency": generate_model_diagnostic(
                base_outputs["tours"],
                proj_outputs["tours"],
                "stop_frequency",
                selected_columns,
            ),
        },
        lazy=True,
    )

    trip_choices = mo.accordion(
        {
            "#### trip_purpose": generate_model_diagnostic(
                base_outputs["trips"],
                proj_outputs["trips"],
                "purpose",
                selected_columns,
            ),
            # "#### trip_departure_choice": generate_model_diagnostic(
            #     base_outputs["trips"],
            #     proj_outputs["trips"],
            #     "primary_purpose",
            #     selected_columns,
            # ),
            "#### trip_scheduling_choice": mo.vstack(
                [
                    generate_model_diagnostic(
                        base_outputs["tours"],
                        proj_outputs["tours"],
                        col,
                        selected_columns,
                    )
                    for col in [
                        "outbound_duration",
                        "main_leg_duration",
                        "inbound_duration",
                    ]
                ]
            ),
            "#### trip_departure_choice": generate_model_diagnostic(
                base_outputs["trips"],
                proj_outputs["trips"],
                "depart",
                selected_columns,
            ),
            "#### trip_mode_choice": generate_model_diagnostic(
                base_outputs["trips"],
                proj_outputs["trips"],
                "trip_mode",
                selected_columns,
            ),
        },
        lazy=True,
    )


    models_tab_ui = mo.vstack(
        [
            mo.accordion({"### Long-term choices": long_term_choices}),
            mo.accordion({"### Tours": tour_choices}),
            mo.accordion({"### Trips": trip_choices}),
            # mo.hstack(
            #     [
            #         mo.image(
            #             "https://rsginc.com/wp-content/uploads/2021/01/Example-Activity-Based-Model-and-Submodel-Structure-2048x1907.png",
            #             width=650,
            #             caption="ActivitySim model structure",
            #         )
            #     ],
            #     justify="center",
            # ),
        ]
    )
    return long_term_choices, models_tab_ui, tour_choices, trip_choices


@app.cell
def _():
    # base_outputs["tours"].head().collect()
    # base_outputs["tours"].group_by("composition").len().collect()
    return


@app.cell
def _():
    # (
    #     proj_outputs["tours"]
    #     # .filter(pl.col("tour_category") == "atwork")
    #     .filter(~pl.col("atwork_subtour_frequency").is_in(["", "no_subtours"]))
    # ).collect()
    return


@app.cell
def column_filter_table(base_outputs, pl):
    all_columns_df = pl.concat(
        [
            pl.DataFrame(
                {
                    "table": table_name,
                    "variable": base_outputs[table_name].collect_schema().keys(),
                }
            )
            for table_name in [
                key
                for key in base_outputs.keys()
                if key not in ["skims", "land_use"]
            ]
        ]
    ).filter(~pl.col("variable").str.ends_with("_id"))
    return (all_columns_df,)


@app.cell
def _(all_columns_df, mo):
    column_table = mo.ui.table(all_columns_df.to_pandas())
    return (column_table,)


@app.cell
def _(column_table):
    # selected_columns = None

    selected_columns = (
        column_table.value["variable"].to_list()
        if len(column_table.value["variable"]) > 0
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


@app.cell
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
            "skims": "skims.parquet",
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
def _(Any, Dict, Optional, base_outputs, mo, pl, proj_outputs):
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
            # "one-car_households": outputs["households"].pipe(
            #     _total_catagorical_variable, "AUTO_OWNERSHIP", 1
            # ),
            # "two-car_households": outputs["households"].pipe(
            #     _total_catagorical_variable, "AUTO_OWNERSHIP", 2
            # ),
            # "three_or_more cars_households": outputs["households"].pipe(
            #     _total_catagorical_variable, "AUTO_OWNERSHIP", 2
            # ),
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
        direction = _get_direction(proj_value, base_value)

        # Calculate the percentage difference.
        pct_diff = ((proj_value / base_value) - 1) * 100

        return mo.stat(
            label=formatted_label,
            value=proj_value,
            caption=f"{pct_diff:.1f}% (Base: {base_value:,.{0 if base_value == int(base_value) else 2}f})",
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
    return (summary_cards,)


@app.cell
def _(GT, List, Optional, cs, mo, pl, px, scenario_discrete_color_map):
    def generate_model_diagnostic(
        base: pl.LazyFrame,
        proj: pl.LazyFrame,
        variable: str,
        by_columns: Optional[List[str]] = None,
    ):
        """
        Generate diagnostic visuals and table comparing aggregated Base and Project scenarios.

        Parameters:
            base (pl.LazyFrame): The base scenario data.
            proj (pl.LazyFrame): The project scenario data.
            variable (str): Primary variable to aggregate by.
            by_columns (Optional[List[str]]): Additional grouping columns.

        Returns:
            A vertically stacked object combining plotly tabs and a formatted table.
        """

        def _compute_agg_by(
            lazy_df: pl.LazyFrame, group_cols: List[str]
        ) -> pl.DataFrame:
            return (
                lazy_df.group_by(group_cols)
                .agg(len=pl.len().cast(pl.Int64))
                .collect()
            )

        def _gen_great_table(tab_df: pl.DataFrame):
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
                    na_color="lightgray",
                )
                .data_color(
                    columns=["pct_diff"],
                    palette="RdYlBu",
                    domain=[-1, 1],
                    na_color="lightgray",
                )
            )

        def _gen_fig(col: str):
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
                facet_col=existed_by_columns[0] if existed_by_columns else None,
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
            )
            if col == "share":
                fig.update_layout(yaxis=dict(tickformat=".0%"))

            # Remove individual x-axis titles from all facets
            fig.for_each_xaxis(lambda axis: axis.update(title_text=""))

            # Add a global x-axis label as an annotation (position may need adjustment)
            fig.add_annotation(
                text=variable,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.1,  # x centered; y=0 aligns with the bottom of the figure
                showarrow=False,
                font=dict(size=14),
                xanchor="center",
                yanchor="top",
            )
            return fig

        # Filter the grouping columns to those that exist in 'base'
        existed_by_columns = (
            [col for col in by_columns if col in base.collect_schema().keys()]
            if by_columns
            else None
        )
        agg_cols = [variable] + (existed_by_columns if existed_by_columns else [])

        # Compute aggregated data for Base and Project scenarios
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

        # Add share column (grouped by scenario)
        agg_df = agg_df.with_columns(
            share=pl.col("len") / pl.col("len").sum().over("scenario")
        )

        # Pivot data to compare Base and Project counts side by side
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

        # Combine tabs and table using mo.vstack and mo.ui.tabs (assumed external modules)
        return mo.vstack(
            [
                mo.ui.tabs({"Share": _gen_fig("share"), "Count": _gen_fig("len")}),
                _gen_great_table(agg_df_pivoted),
            ]
        )
    return (generate_model_diagnostic,)


@app.cell
def _(mo):
    mo.hstack(
        [mo.md(rf"""# {mo.icon("lucide:file-user")} Population""")],
        justify="start",
    )
    return


@app.cell
def _(base_outputs, compare_distributions, mo, proj_outputs):
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
                    compare_distributions(
                                base_outputs, proj_outputs, "persons", "OCCUPATION"
                    )
                ]
            ),
            "### Households": mo.hstack(
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
        },
        lazy=True,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""-------""")
    return


@app.cell
def _(mo):
    mo.hstack([mo.md(rf"""# {mo.icon("lucide:repeat")} Tours""")], justify="start")
    return


@app.cell
def _(cutoff_distance_slider):
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
def _(pl):
    def gen_plot_dist_to_mandatory_destinations(
        scenario_outputs, tour_type, scenario_name
    ):
        return (
            scenario_outputs["tours"]
            .filter(
                pl.col("tour_category") == "mandatory",
                pl.col("tour_type") == tour_type,
            )
            .join(
                scenario_outputs["skims"].with_columns(
                    pl.col("origin", "destination").cast(pl.Float64),
                    pl.col("SOV_FREE_DISTANCE__AM").alias("distance"),
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
            # .select("distance")
            .collect()
        )
    return (gen_plot_dist_to_mandatory_destinations,)


@app.cell
def _(dist_business_tours, dist_school_tours, dist_work_tours, mo):
    cutoff_distance_slider = mo.ui.slider(
        start=0,
        stop=max(
            [
                dist_work_tours["distance"].max(),
                dist_business_tours["distance"].max(),
                dist_school_tours["distance"].max(),
            ]
        ),
        label="Cut-off distance (KM)",
        value=80,
    )
    return (cutoff_distance_slider,)


@app.cell
def _(
    base_outputs,
    gen_plot_dist_to_mandatory_destinations,
    pl,
    proj_outputs,
):
    _tour_type = "school"

    _base_dist = gen_plot_dist_to_mandatory_destinations(
        base_outputs, _tour_type, "Base"
    )
    _proj_dist = gen_plot_dist_to_mandatory_destinations(
        proj_outputs, _tour_type, "Project"
    )

    dist_school_tours = pl.concat([_base_dist, _proj_dist])
    return (dist_school_tours,)


@app.cell
def _(
    base_outputs,
    gen_plot_dist_to_mandatory_destinations,
    pl,
    proj_outputs,
):
    _tour_type = "business"

    _base_dist = gen_plot_dist_to_mandatory_destinations(
        base_outputs, _tour_type, "Base"
    )
    _proj_dist = gen_plot_dist_to_mandatory_destinations(
        proj_outputs, _tour_type, "Project"
    )

    dist_business_tours = pl.concat([_base_dist, _proj_dist])
    return (dist_business_tours,)


@app.cell
def _(
    base_outputs,
    gen_plot_dist_to_mandatory_destinations,
    pl,
    proj_outputs,
):
    _tour_type = "work"

    _base_dist = gen_plot_dist_to_mandatory_destinations(
        base_outputs, _tour_type, "Base"
    )
    _proj_dist = gen_plot_dist_to_mandatory_destinations(
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
    mo.accordion({"## Plots": mo.vstack(
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
        )}, lazy=True
    )
    return


if __name__ == "__main__":
    app.run()
