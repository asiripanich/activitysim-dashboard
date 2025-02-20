# ActivitySim dashboard

![](images/banner.png)

[![ci](https://github.com/asiripanich/activitysim-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/asiripanich/activitysim-dashboard/actions/workflows/ci.yml)

## Get started

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
