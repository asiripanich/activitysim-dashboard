# ActivitySim dashboard 
[![ci](https://github.com/asiripanich/activitysim-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/asiripanich/activitysim-dashboard/actions/workflows/ci.yml)
[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-yellow.svg)](https://www.tidyverse.org/lifecycle/#experimental)
<p align="center">
<img src="https://github.com/user-attachments/assets/09a1ef48-d946-4b85-bd8d-fa94a625fb3f" width="500" alt="asim-dashboard">
</p>

The ActivitySim Dashboard is a high-performance, memory-efficient, and extensible Python dashboard for validating and comparing two runs of ActivitySim. It leverages:

- Parquet output tables (introduced in ActivitySim v1.3)
- Polars LazyFrame for reactive queries that keep you moving without long waits
- Marimo for a shareable, reproducible, and git-friendly interface

**✨ Key Features**

- Side-by-side comparison of base vs. project model runs
- Flexible sub-model configuration via a TOML file
- Fast data loading and filtering with Polars
- Interactive UI powered by Marimo



## Why another dashboard for ActivitySim?

While ActivitySim’s [documentation](https://activitysim.github.io/activitysim/v1.3.1/users-guide/visualization.html) recommends SimWrapper, it currently lacks built‑in support for computing and visualizing differences between model runs. This dashboard fills that gap until a community standard emerges.

## Get started

### `dashboard.py`

To get started, you only need to install two Python packages: 

- [uv](https://docs.astral.sh/uv/): "*A Python package and project manager*"
- [marimo](https://marimo.io): "*an open-source reactive notebook for Python — reproducible, git-friendly, executable as a script, and shareable as an app.*" 

I recommend installing `uv` using one of the installation methods [here](https://docs.astral.sh/uv/getting-started/installation/#installation-methods). Once you have `uv` on your machine, install `marimo` as a [`uv tool`](https://docs.astral.sh/uv/concepts/tools/).


Then, install marimo as a tool in uv:

```sh
uv tool install marimo
```

#### Run the dashboard

Edit mode (interactive):

```sh
uvx marimo edit --sandbox dashboard.py
```

Read‑only mode (view only):

```sh
uvx marimo run --sandbox dashboard.py
```

If using a Conda/virtual environment:

```sh
marimo edit --sandbox dashboard.py
```

#### How to use the dashboard with your ActivitySim outputs?

It is common that different ActivitySim implementations may use slighly different field names or have a different set of sub-models (e.g., some have an autonomous vehicle ownership model while others don't). To make the dashboard more flexible, we use a TOML configuration file to define the sub-models to show on the dashboard. The default configuration file is `dashboard-config.toml`, which you can modify to suit your implementation.

Let's start from the basics. The below is the first section of `dashboard-config.toml` defines the title of the dashboard and the default directories for the base and project model runs. The directories can be relative paths to the dashboard script or absolute paths. They are just the default options and can be changed in the dashboard.

```toml
[dashboard]
# the title of the dashboard
title = "Example ActivitySim dashboard"

[base]
# the default directory of the base model run
directory = "example_data/mtc/base"

[project]
# the default directory of the project model run
directory = "example_data/mtc/project"
```

Models are grouped into three sections. You can place any sub-model under any category.

- `household_person`: household or person-level decisions
- `tour`: tour-level decisions
- `trip`: trip-level decisions

```toml
[models.household_person]
# e.g., work_from_home, auto_ownership

[models.tour]
# e.g., tour_mode_choice

[models.trip]
# e.g., trip_purpose
```

#### Example: At-work Tour Mode Choice

To understand the building blocks of a sub-model in the dashboard, let's take a look at the `atwork_subtour_mode_choice` sub-model as an example. As `atwork_subtour_mode_choice` is a tour-level decision, it is placed under `models.atwork_subtour_mode_choice`. The sub-model configuration is defined as follows:

```toml
[models.tour.atwork_subtour_mode_choice]
table = "tours"
result_field = "tour_mode"
filter_expr = "SELECT * FROM self WHERE tour_category == 'atwork'"
```

Here is what each field means:

- The `table` field indicates the table that contains the results of the sub-model. Here are the tables that can be used: `persons`, `households`, `tours`, `trips`, `joint_tour_participants`, `skims`, `zones`, and `land_use`.
- The `result_field` field indicates the field that contains the results of the sub-model. 
- The `filter_expr` field is a SQL statement that filters `table`. This is passed to the `sql` method of the Polars DataFrame.

For `atwork_subtour_mode_choice`, the table that contains the results is `tours`, and the field that contains the results is `tour_mode`. The filter expression filters the `tour` table to only include tours that are classified as `atwork`. As you can imagine, if you remove the `filter_expr` field from the configuration, you would get the mode choice results of all the tours, not just for at-work subtours. Hence, this approach provides a flexible way to display the results of the sub-models across all the final output tables from ActivitySim.

#### Example: Arbitrary Column Comparison

Why limit yourself to sub‑model outputs? Thanks to the configuration file’s flexibility, you can compare any column in your tables. For example, to compare `ptype` (a person category field in the `persons` table created by the `annotate_persons` step), simply add:

```toml
[models.household_person.ptype]
table = "persons"
result_field = "ptype"
```

This configuration will add a new section under `households_persons` inside the dashboard called `ptype` that compares the `ptype` field between the base and project scenarios.

#### Example: Location Models

Location models work a bit differently than the other sub-models, hence they deserve some special fields to allow us to investigate their results more closely. 

Let's take a look at the `work_location` sub-model under `household_person` as an example. As `work_location` is a person-level decision, it is placed under `models.household_person`.

```toml
[models.household_person.work_location]
table = "persons"
result_field = "workplace_zone_id"
filter_expr = "SELECT * FROM self WHERE workplace_zone_id > 0"
skims_variable = "DIST"
land_use_control_variable = "TOTEMP"
origin_zone_variable = "home_zone_id"
```

The first three fields have been introduced to you in the previous example. 

- `skims_variable` is the field in the `skims` table (converted from skims.omx via input_converter.py) that will be used to compare between the base and project scenarios. This could be a distance field or another travel impedance field.
- The `land_use_control_variable` field is the control value (e.g. total employment) in the `land_use` table. This is compared against the simulated scenario results - not between the base and project scenarios.
- The `origin_zone_variable` field is the zone ID in your main table (`table`) that marks the origin location for each record, used to look up the corresponding `skims_variable` to the destination (`result_field`).

For the work_location example:

- `result_field` is `workplace_zone_id` in the `persons` table.
- `filter_expr` (`WHERE workplace_zone_id > 0`) restricts to people who are workers.
- `skims_variable` is the network distance measure in the `skims` table.
- `land_use_control_variable` is the total employment count in the `land_use` table.
- `origin_zone_variable` is each person’s `home_zone_id`.


The dashboard currently offers two ways to visualise location model results:

1.	Compare each scenario’s output against its `land_use_control_variable`.
2.	Compare `skims_variable` values between the base and project scenarios.

### `input_converter.py`

This is another Marimo notebook that helps convert CSV, SHP, GeoJSON and OMX files to the Parquet format, enabling you to create compatible inputs for `dashboard.py`. This is particularly useful if you want to use the dashboard with ActivitySim final outputs provided in CSV format.
