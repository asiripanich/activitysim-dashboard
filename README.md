# ActivitySim dashboard

![](images/banner.png)

[![ci](https://github.com/asiripanich/activitysim-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/asiripanich/activitysim-dashboard/actions/workflows/ci.yml)

## Get started


### Install required dependencies

To get started, you only need to install two Python packages: 

- [uv](https://docs.astral.sh/uv/): "*A Python package and project manager*"
- [marimo](https://marimo.io): "*an open-source reactive notebook for Python — reproducible, git-friendly, executable as a script, and shareable as an app.*" 

### Running the dashboard

```sh
# using an environment e.g., Conda or Python virtual environment
marimo edit --sandbox dashboard.py

# using a standalone uv
uvx marimo edit --sandbox dashboard.py
```
