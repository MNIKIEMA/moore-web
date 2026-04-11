# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
# ]
# ///
"""Marimo notebook — explore and visualise an AlignedCorpus JSON.

Run with:
    uv run marimo edit notebooks/explore_aligned.py
    uv run marimo run  notebooks/explore_aligned.py   # read-only / shareable
"""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium", app_title="Aligned Corpus Explorer")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import pandas as pd

    return Path, mo, mticker, pd, plt


@app.cell
def _(mo):
    file_path_input = mo.ui.text(
        value="final_data_hf/conseils_ministres_aligned.jsonl",
        label="Aligned JSON path",
        placeholder="aligned.jsonl",
        full_width=True,
    )
    file_path_input
    return (file_path_input,)


@app.cell
def _(Path, file_path_input, mo, pd):
    _path = Path(file_path_input.value)
    print(Path.cwd())
    if not _path.exists():
        mo.stop(True, mo.callout(mo.md(f"File **{_path}** not found."), kind="danger"))

    lines = _path.suffix == ".jsonl"

    df = pd.read_json(_path, lines=lines)
    source = "unknown"
    return df, source


@app.cell
def _(df, mo, source):
    _sc = df["laser_score"]
    mo.hstack(
        [
            mo.stat(label="Pairs", value=str(len(df)), bordered=True),
            mo.stat(label="Source", value=source, bordered=True),
            mo.stat(label="Mean score", value=f"{_sc.mean():.3f}", bordered=True),
            mo.stat(label="Median score", value=f"{_sc.median():.3f}", bordered=True),
            mo.stat(label="Min score", value=f"{_sc.min():.3f}", bordered=True),
            mo.stat(label="Max score", value=f"{_sc.max():.3f}", bordered=True),
        ],
        justify="start",
    )
    return


@app.cell
def _(df, mo):
    _sc = df["laser_score"]
    threshold_slider = mo.ui.slider(
        start=float(_sc.min()),
        stop=float(_sc.max()),
        step=0.01,
        value=round(float(_sc.median()), 2),
        label="Min score threshold",
        show_value=True,
        full_width=True,
    )
    threshold_slider
    return (threshold_slider,)


@app.cell
def _(df, mo, mticker, plt, threshold_slider):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(df["laser_score"], bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(
        threshold_slider.value,
        color="#DD4444",
        linewidth=1.8,
        linestyle="--",
        label=f"threshold = {threshold_slider.value:.2f}",
    )
    ax.set_xlabel("Cosine similarity score", fontsize=11)
    ax.set_ylabel("Pairs", fontsize=11)
    ax.set_title("Score distribution", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=10)
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(df, mo, plt, threshold_slider):
    _filtered = df[df["laser_score"] >= threshold_slider.value]

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sc2 = ax2.lines(
        _filtered["len_ratio"],
        c=_filtered["laser_score"],
        cmap="RdYlGn",
        alpha=0.55,
        s=18,
        linewidths=0,
    )
    plt.colorbar(sc2, ax=ax2, label="Score")
    ax2.set_xlabel("French sentence length (chars)", fontsize=10)
    ax2.set_ylabel("Mooré sentence length (chars)", fontsize=10)
    ax2.set_title(
        f"Sentence lengths — {len(_filtered)} pairs (score ≥ {threshold_slider.value:.2f})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    mo.mpl.interactive(fig2)
    return


@app.cell
def _(df, mo, pd):
    _pcts = [10, 25, 50, 75, 90, 95, 99]
    _rows = [{"Percentile": f"p{p}", "Score": f"{df['laser_score'].quantile(p / 100):.4f}"} for p in _pcts]
    mo.md("### Score percentiles"), mo.ui.table(pd.DataFrame(_rows), pagination=False, selection=None)
    return


@app.cell
def _(df, mo, threshold_slider):
    _filtered = (
        df[df["laser_score"] >= threshold_slider.value]
        .sort_values("laser_score", ascending=False)
        .reset_index(drop=True)
    )
    _display = _filtered[["laser_score", "french", "moore"]].copy()
    _display["laser_score"] = _display["laser_score"].map("{:.4f}".format)

    mo.vstack(
        [
            mo.md(
                f"### Pair browser  —  **{len(_filtered)}** pairs with score ≥ {threshold_slider.value:.2f}"
                f"  ({100 * len(_filtered) / len(df):.1f}% of total)"
            ),
            mo.ui.table(
                _display,
                pagination=True,
                page_size=20,
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(df, mo):
    _cutoff = df["laser_score"].quantile(0.10)
    _low = (
        df[df["laser_score"] < _cutoff]
        .sort_values("laser_score")
        .reset_index(drop=True)[["laser_score", "french", "moore", "comet_qe"]]
        .copy()
    )
    _low["laser_score"] = _low["laser_score"].map("{:.4f}".format)
    _low["comet_qe"] = _low["comet_qe"].map("{:.4f}".format)

    mo.vstack(
        [
            mo.md(f"### Bottom 10% — potential misalignments  (score < {_cutoff:.3f})"),
            mo.ui.table(_low, pagination=True, page_size=10, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
