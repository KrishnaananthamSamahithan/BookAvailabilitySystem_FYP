import ast
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "Flight_Bookability_Analysis_V2.ipynb"


SECTIONS = [
    {
        "title": "Configuration",
        "summary": (
            "This module defines the central project configuration, path management, "
            "taxonomy settings, and the documented prediction-time assumption."
        ),
        "path": "src/config.py",
    },
    {
        "title": "Utility Helpers",
        "summary": (
            "Shared helper functions used across schema handling, URL parsing, numeric "
            "conversion, text normalization, and artifact writing."
        ),
        "path": "src/utils.py",
    },
    {
        "title": "Schema Resolution",
        "summary": (
            "Alias-based schema validation ensures the pipeline does not touch raw "
            "columns before confirming the logical fields required by the project."
        ),
        "path": "src/schema.py",
    },
    {
        "title": "Canonical Label Engineering",
        "summary": (
            "This module maps noisy operational statuses into the final coherent target "
            "taxonomy used throughout the final project."
        ),
        "path": "src/labels.py",
    },
    {
        "title": "Preprocessing Pipeline",
        "summary": (
            "This module performs the corrected preprocessing flow: schema-safe field "
            "resolution, timestamp parsing, audit logging, deduplication, and leakage-free "
            "base dataset construction."
        ),
        "path": "src/preprocessing.py",
    },
    {
        "title": "Feature Engineering",
        "summary": (
            "This module builds only valid feature families: topology, temporal context, "
            "real-price features when available, and prior-only reliability/history signals."
        ),
        "path": "src/features.py",
    },
    {
        "title": "Temporal Splitting",
        "summary": (
            "Chronological train, validation, and test splitting plus rolling backtest "
            "window generation live here."
        ),
        "path": "src/split.py",
    },
    {
        "title": "Evaluation Metrics",
        "summary": (
            "This module implements the required research metrics including macro F1, "
            "weighted F1, balanced accuracy, log loss, multiclass Brier, confusion matrix, "
            "and classwise ECE."
        ),
        "path": "src/metrics.py",
    },
    {
        "title": "Model Comparison",
        "summary": (
            "This module defines the candidate model families and the transparent model "
            "selection rule used on the validation split."
        ),
        "path": "src/models.py",
    },
    {
        "title": "Calibration Comparison",
        "summary": (
            "This module implements the required calibration strategies: no calibration, "
            "temperature scaling, one-vs-rest isotonic, and multinomial logistic calibration."
        ),
        "path": "src/calibration.py",
    },
    {
        "title": "Evaluation Helpers",
        "summary": (
            "Small reporting helpers for converting metric dictionaries into comparison "
            "tables that can be saved as report artifacts."
        ),
        "path": "src/evaluation.py",
    },
    {
        "title": "Reporting And Diagnostics",
        "summary": (
            "This module generates dissertation-facing diagnostics such as subgroup "
            "performance, calibration parity, and schema resolution reporting."
        ),
        "path": "src/reporting.py",
    },
    {
        "title": "Operational Policy Simulation",
        "summary": (
            "This module contains the suppression policy simulation and the reranking proxy "
            "used to connect calibrated probabilities to metasearch decisions."
        ),
        "path": "src/simulation.py",
    },
    {
        "title": "Main Training Pipeline",
        "summary": (
            "This is the final end-to-end training entry point. It orchestrates preprocessing, "
            "feature construction, temporal splitting, model comparison, calibration selection, "
            "evaluation, ablation studies, rolling backtests, policy simulation, and artifact saving."
        ),
        "path": "src/train.py",
    },
    {
        "title": "Inference Pipeline",
        "summary": (
            "This module applies the saved model bundle to new CSV data using the final "
            "preprocessing and feature contract."
        ),
        "path": "src/inference.py",
    },
]


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def line_slice(source: str, start: int, end: int) -> str:
    lines = source.splitlines()
    return "\n".join(lines[start - 1 : end]).rstrip() + "\n"


def get_segments(source: str):
    tree = ast.parse(source)
    body = tree.body
    segments = []

    if not body:
        return segments

    def node_end(index: int) -> int:
        node = body[index]
        end = node.end_lineno
        next_start = body[index + 1].lineno if index + 1 < len(body) else None
        if next_start is not None and end >= next_start:
            end = next_start - 1
        return end

    def is_named_block(node) -> bool:
        return isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If),
        )

    index = 0
    while index < len(body):
        node = body[index]

        if not is_named_block(node):
            start = node.lineno
            end = node_end(index)
            index += 1

            while index < len(body) and not is_named_block(body[index]):
                end = node_end(index)
                index += 1

            label = "Imports, constants, and top-level configuration" if not segments else "Top-level execution block"
            kind = "imports_and_globals" if not segments else "block"
            segments.append((kind, start, end, label))
            continue

        start = node.lineno
        end = node_end(index)

        if isinstance(node, ast.FunctionDef):
            label = f"Function: `{node.name}`"
        elif isinstance(node, ast.AsyncFunctionDef):
            label = f"Async Function: `{node.name}`"
        elif isinstance(node, ast.ClassDef):
            label = f"Class: `{node.name}`"
        else:
            test_text = ast.unparse(node.test) if hasattr(ast, "unparse") else "entrypoint"
            label = f"Conditional Block: `{test_text}`"

        segments.append(("block", start, end, label))
        index += 1

    return segments


def intro_cells():
    intro_md = """# Offline Multi-Class Flight Bookability Prediction (Colab Review Notebook)

**Project Goal:** Build and evaluate an offline or batch multiclass flight bookability prediction pipeline for metasearch offers.

This notebook is organized as an examiner-friendly code walkthrough for Google Colab. The code is broken into granular sections so the implementation can be read, downloaded, and reviewed comfortably without needing to jump around the repository.

### What your mentor can review here
- schema validation and label engineering
- corrected preprocessing logic
- valid feature engineering only
- leakage-safe temporal splitting
- model comparison and calibrator comparison
- evaluation metrics and reliability diagnostics
- ablation support and operational policy simulation
- final training and inference entry points
"""

    colab_md = """## Colab Setup

If this notebook is opened in Google Colab, the following optional setup cell can be used to install the main dependencies. It is safe to skip if the notebook is being reviewed rather than executed.
"""

    colab_code = """# Optional Colab setup
# Uncomment this cell if you want to run the code in Google Colab.

# !pip install pandas numpy scikit-learn catboost matplotlib seaborn streamlit nbformat
"""

    notes_md = """## Review Notes

- The notebook mirrors the current repository state as of generation time.
- `src/` contains the main implementation.
- `src/` now contains the final coherent pipeline.
- Older prototype code is intentionally not included in this notebook because it is no longer the authoritative path.
- Some scripts expect local project files such as `data/raw/...`, `data/processed/...`, and `artifacts/...`.
- The notebook is structured for review first and execution second.
"""

    return [
        nbf.v4.new_markdown_cell(intro_md),
        nbf.v4.new_markdown_cell(colab_md),
        nbf.v4.new_code_cell(colab_code),
        nbf.v4.new_markdown_cell(notes_md),
    ]


def section_cells(title: str, summary: str, relative_path: str):
    source = read_source(relative_path)
    segments = get_segments(source)
    cells = [
        nbf.v4.new_markdown_cell(
            f"## {title}\n\n**Source file:** `{relative_path}`\n\n{summary}"
        )
    ]

    for kind, start, end, label in segments:
        code = line_slice(source, start, end)
        if not code.strip():
            continue

        if kind == "imports_and_globals":
            description = (
                f"### {label}\n\n"
                f"This block contains the imports, module-level constants, and shared state used by "
                f"`{relative_path}`."
            )
        else:
            description = (
                f"### {label}\n\n"
                f"This block is extracted directly from `{relative_path}` so the implementation can be "
                f"reviewed in smaller steps."
            )

        cells.append(nbf.v4.new_markdown_cell(description))
        cells.append(nbf.v4.new_code_cell(code))

    return cells


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
            "colab": {
                "name": "Flight_Bookability_Analysis_V2.ipynb",
                "provenance": [],
            },
        }
    )

    cells = intro_cells()
    for section in SECTIONS:
        cells.extend(
            section_cells(
                section["title"],
                section["summary"],
                section["path"],
            )
        )

    nb.cells = cells
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        nbf.write(nb, handle)

    print(f"Notebook compiled successfully at {NOTEBOOK_PATH}")
    print(f"Total cells written: {len(nb.cells)}")


if __name__ == "__main__":
    build_notebook()
