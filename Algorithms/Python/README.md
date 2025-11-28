# cmo-py

[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Optimisation algorithm implementations in Python

## Installation

<details>

<summary>Clone the repository</summary>

```shell
git clone https://github.com/AshrithSagar/E0230-CMO-2025.git
cd E0230-CMO-2025
```

</details>

<details>

<summary>Install uv</summary>

Install [`uv`](https://docs.astral.sh/uv/), if not already.
Check [here](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

It is recommended to use `uv`, as it will automatically install the dependencies in a virtual environment.
If you don't want to use `uv`, skip to the next step.

**TL;DR: Just run**

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

<details>

<summary>Install the package</summary>

The dependencies are listed in the [pyproject.toml](pyproject.toml) file.

Install the package in editable mode (recommended):

```shell
# Using uv
uv sync

# Or with pip
pip3 install -e .
```

</details>

## License

This project falls under the [MIT License](../../LICENSE).
