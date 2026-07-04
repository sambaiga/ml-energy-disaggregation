# Machine Learning for Energy Disaggregation

Source for **[Machine Learning for Energy Disaggregation](https://sambaiga.github.io/ml-energy-disaggregation/)**, a structured, project-based path through non-intrusive load monitoring (NILM): from raw smart-meter signal to appliance-level insight, built to be run and not just read.

Every chapter is a Jupyter notebook or QMD file meant to be run top to bottom, not just read. The book is built with [Quarto](https://quarto.org).

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/sambaiga/ml-energy-disaggregation
   cd ml-energy-disaggregation
   ```

2. **Install [uv](https://docs.astral.sh/uv/getting-started/installation/)** (if not already installed)

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**

   ```bash
   uv sync --extra modelling --extra dev --extra test
   ```

4. **Register the Jupyter kernel** (needed for notebooks and for `quarto render`)

   ```bash
   uv run python -m ipykernel install --user --name ark --display-name "ark"
   ```

5. **Install pre-commit hooks**

   ```bash
   uv run pre-commit install
   uv run pre-commit install --hook-type pre-push
   ```

6. **Install [Quarto](https://quarto.org/docs/get-started/)**, then render the book locally

   ```bash
   uv run quarto preview
   ```

## Project structure

```
tutorials/            # chapters, one notebook or .qmd per chapter, grouped into parts
tutorials/_quarto/    # brand.scss, the site's visual theme
ark/plot/             # shared branding module: theme, tokens, and Great Tables styling
                       # used by every chart and table in the book
tests/                # tests for any code promoted out of a notebook into ark/
_quarto.yml           # book structure: parts, chapters, navigation, theme
index.qmd             # landing page
preface.qmd           # why this book
about.qmd             # about the author
```

## Contributing

Each chapter moves through a feature branch and a pull request before merging to `main`. CI runs on every PR:

- `ARK CID` (`.github/workflows/check_pullrequest.yaml`): pre-commit hooks, ruff lint/format, tests with coverage, and a structure-only Quarto render.
- `Check Notebooks` (`.github/workflows/check-notebooks.yml`): executes changed notebooks end to end via `nbmake`.
- `Check QMD Files` (`.github/workflows/check-qmd.yml`): lints `.qmd` files for em-dashes, invalid code-block syntax, and broken cross-references.

Merges to `main` trigger `Publish Book` (`.github/workflows/publish-book.yml`), which renders the full site and deploys it to GitHub Pages.

## License

Code is licensed under [Apache 2.0](LICENSE).
