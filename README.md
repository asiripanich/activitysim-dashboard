# ActivitySim dashboard 
[![ci](https://github.com/asiripanich/activitysim-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/asiripanich/activitysim-dashboard/actions/workflows/ci.yml)
[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-yellow.svg)](https://www.tidyverse.org/lifecycle/#experimental)
<p align="center">
<img src="https://github.com/user-attachments/assets/09a1ef48-d946-4b85-bd8d-fa94a625fb3f" width="500" alt="asim-dashboard">
</p>

A high-performance, memory-efficient, reproducible, and extensible ActivitySim dashboard written in Python. The goal is to build a dashboard that allows ActivitySim developers to easily validate their model and compare its runs. This dashboard utilises the Parquet format—a recent addition to the output table format in ActivitySim v1.3—and Polars’ [Lazy DataFrame](https://docs.pola.rs/user-guide/lazy/) to achieve a reactive experience that doesn’t make you wait until your coffee gets cold.

## Get started

### `dashboard.py`

To get started, you only need to install two Python packages: 

- [uv](https://docs.astral.sh/uv/): "*A Python package and project manager*"
- [marimo](https://marimo.io): "*an open-source reactive notebook for Python — reproducible, git-friendly, executable as a script, and shareable as an app.*" 

I recommend installing `uv` using one of the installation methods [here](https://docs.astral.sh/uv/getting-started/installation/#installation-methods). Once you have `uv` on your machine, install `marimo` as a [`uv tool`](https://docs.astral.sh/uv/concepts/tools/).

```sh
# install marimo as a tool in uv
uv tool install marimo

# run the dashboard using marimo uv tool
uvx marimo edit --sandbox dashboard.py

# for read-only mode
uvx marimo run --sandbox dashboard.py

# using an environment e.g., Conda or Python virtual environment
marimo edit --sandbox dashboard.py
```

### `input_converter.py`

This is another Marimo notebook that helps convert CSV and OMX files to the Parquet format, enabling you to create compatible inputs for `dashboard.py`. This is particularly useful if you want to use the dashboard with ActivitySim final outputs provided in CSV format.
