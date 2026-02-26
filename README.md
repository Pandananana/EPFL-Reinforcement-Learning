# RL Notebook Workflow

This repository uses `uv` for Python dependency management and `pre-commit` to keep Jupyter notebooks clean in Git.

## Install uv

Install `uv` by following the official Astral docs:

- https://docs.astral.sh/uv/getting-started/installation/

## Project setup

From the repo root, install dependencies:

```bash
uv sync
```

## Install the pre-commit hook

Install git hooks for this repo:

```bash
uv run pre-commit install
```

## What the hook is for

On each commit, the notebook hook runs:

```bash
uv run nbstripout --keep-output --extra-keys "metadata.widgets cell.metadata.execution output.metadata output.execution_count"
```

This keeps notebook outputs (plots/text results) but removes noisy fields that create unnecessary diff churn:

- execution counters and transient execution metadata
- widget state metadata
- output metadata fields that often change run-to-run
