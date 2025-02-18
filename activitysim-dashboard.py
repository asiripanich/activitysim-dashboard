# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair==5.5.0",
#     "folium==0.19.4",
#     "geopandas==1.0.1",
#     "ipyleaflet==0.19.2",
#     "ipywidgets==8.1.5",
#     "keplergl==0.3.7",
#     "leafmap==0.42.9",
#     "lonboard==0.10.4",
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

__generated_with = "0.11.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import polars as pl
    import altair as alt
    from typing import Any, Dict, Optional
    import plotly.express as px
    import geopandas as gpd
    import omx

    import leafmap.foliumap as leafmap
    import folium
    return (
        Any,
        Dict,
        Optional,
        alt,
        folium,
        gpd,
        leafmap,
        mo,
        omx,
        os,
        pl,
        px,
    )


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.image(
                src="https://research.ampo.org/wp-content/uploads/2024/07/activitysim_logo_light.jpg",
                height=100,
            ),
            mo.md("# ActivitySim dashboard"),
        ]
    )
    return


@app.cell
def _(mo):
    base_label = mo.ui.text(
        placeholder="Output folder...", label="**Label:** ", value="Base"
    )
    base_dir = mo.ui.text(
        placeholder="Output folder...",
        label="**Folder:** ",
        value=r"/Users/amarin/GitHub/tmp/activitysim-prototype-mtc/output_base",
        full_width=True,
    )

    proj_label = mo.ui.text(
        placeholder="Label...", label="**Label:** ", value="Project"
    )
    proj_dir = mo.ui.text(
        placeholder="Label...",
        label="**Folder:** ",
        value=r"/Users/amarin/GitHub/tmp/activitysim-prototype-mtc/output_project",
        full_width=True,
    )
    return base_dir, base_label, proj_dir, proj_label


@app.cell
def _(base_dir, base_label, mo, proj_dir, proj_label):
    mo.accordion(
        {
            "<h2>Settings</h2>": mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("<h2>Base</h2>"),
                            base_label,
                            base_dir,
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("<h2>Project</h2>"),
                            proj_label,
                            proj_dir,
                        ]
                    ),
                ],
                justify="space-around",
                align="stretch",
                widths="equal",
            )
        },
        lazy=True,
    )
    return


@app.cell
def _(
    base_outputs,
    compare_distributions,
    mo,
    overview_dualmap,
    proj_outputs,
    summary_cards,
):
    mo.ui.tabs(
        {
            # Overview --------------------------------------------------------------------------------
            f"<h2>{mo.icon('lucide:chart-bar-big')} Overview</h2>": mo.vstack(
                [
                    summary_cards,
                    mo.accordion({f"<h2>{mo.icon('lucide:gauge')} Runtime and memory usage</h2>": mo.hstack([])}, lazy=True)
                    # mo.accordion({"<h2>aaa</h2>": mo.hstack([])}, lazy=True)
                    # overview_dualmap,
                ]
            ),
            # Population --------------------------------------------------------------------------------
            f"<h2>{mo.icon('lucide:file-user')} Population</h2>": mo.vstack(
                [
                    mo.md("## Persons"),
                    mo.hstack(
                        [
                            compare_distributions(
                                base_outputs, proj_outputs, "persons", "sex"
                            ),
                            compare_distributions(
                                base_outputs, proj_outputs, "persons", "age"
                            ),
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
                                "auto_ownership",
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
            # Land use --------------------------------------------------------------------------------
            f" <h2>{mo.icon('lucide:earth')} Land use</h2>": mo.vstack(
                [
                    overview_dualmap
                ]
            ),
            # trips --------------------------------------------------------------------------------
            f" <h2>{mo.icon('lucide:route')} Trips</h2>": mo.vstack(
                [
                    mo.hstack(
                        [
                            compare_distributions(
                                base_outputs, proj_outputs, "trips", "trip_mode"
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
            # tours --------------------------------------------------------------------------------
            f"<h2>{mo.icon('lucide:repeat')} Tours</h2>": mo.vstack(
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
        }
    )
    return


@app.cell(hide_code=True)
def _(base_dir, gpd, os, pl, proj_dir):
    def lazy_read_asim_outputs(asim_path):
        # print(os.path.join(asim_path))

        asim_output_names = {
            "persons": "final_persons.parquet",
            "households": "final_households.parquet",
            "trips": "final_trips.parquet",
            "tours": "final_tours.parquet",
            "land_use": "final_land_use.parquet",
        }

        lazyframe_dict = {
            key: pl.scan_parquet(os.path.join(asim_path, value))
            for key, value in asim_output_names.items()
        }

        return lazyframe_dict


    zone_shp = (
        gpd.read_file(
            r"/Users/amarin/GitHub/tmp/activitysim-prototype-mtc/output/summarize/taz1454.geojson",
            use_arrow=True,
        )
        .to_crs(crs="EPSG:4326")
        .rename(columns={"TAZ1454": "zone_id"})
        .filter(items=["zone_id", "geometry"])
    )

    proj_outputs = lazy_read_asim_outputs(proj_dir.value)
    base_outputs = lazy_read_asim_outputs(base_dir.value)

    base_outputs["land_use"] = zone_shp.merge(
        base_outputs["land_use"].collect().to_pandas(), on="zone_id"
    )
    proj_outputs["land_use"] = zone_shp.merge(
        proj_outputs["land_use"].collect().to_pandas(), on="zone_id"
    )
    return base_outputs, lazy_read_asim_outputs, proj_outputs, zone_shp


@app.cell(hide_code=True)
def _(base_outputs, pl, proj_outputs):
    def _compute_summary(outputs):
        def _total_rows(lf):
            return lf.select(pl.len()).collect().item()

        return {
            "total_persons": outputs["persons"].pipe(_total_rows),
            "total_households": outputs["households"].pipe(_total_rows),
            "total_trips": outputs["trips"].pipe(_total_rows),
            "total_tours": outputs["tours"].pipe(_total_rows),
        }


    proj_summary = _compute_summary(proj_outputs)
    base_summary = _compute_summary(base_outputs)
    return base_summary, proj_summary


@app.cell(hide_code=True)
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
            caption=f"{pct_diff:.1f}% (Base: {base_value:,})",
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

        return mo.vstack([mo.hstack(cards, widths="equal", align="center")])


    # Assuming base_summary and proj_summary are defined elsewhere:
    summary_cards = produce_summary_cards(
        {"base": base_summary, "proj": proj_summary}
    )
    return get_direction, produce_card, produce_summary_cards, summary_cards


@app.cell
def _(Dict, List, Optional, Union, pl, px):
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


@app.cell
def _(base_outputs, folium):
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


    overview_dualmap = create_dualmap_leaflet(base_outputs["land_use"])
    return create_dualmap_leaflet, overview_dualmap


if __name__ == "__main__":
    app.run()
