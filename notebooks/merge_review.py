"""Marimo notebook — manually merge/split/reorder mis-segmented FR/MO sentences.

Source-agnostic: it doesn't parse any book itself. It reads a JSONL file
where each line is ``{"<unit_id>": {"fra": [...], "mos": [...]}}`` -- one
line per page/enum/section, sentence-segmented. Any script that produces
that shape can be reviewed here (see scripts/export_review_units.py for the
SIDA book exporter); point the path input below at a different file to
review a different numbered document.

Run with:
    uv run marimo edit notebooks/merge_review.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", app_title="Bilingual Unit Review")


@app.cell
def _():
    import html
    import json
    from pathlib import Path

    import marimo as mo

    return Path, html, json, mo


@app.cell
def _(mo):
    jsonl_path_input = mo.ui.text(
        value="data/review/sida_units.jsonl",
        label="Units JSONL path",
        placeholder="data/review/<name>_units.jsonl",
        full_width=True,
    )
    jsonl_path_input
    return (jsonl_path_input,)


@app.cell
def _(Path, jsonl_path_input, mo):
    JSONL_PATH = Path(jsonl_path_input.value)
    mo.stop(
        not JSONL_PATH.exists(),
        mo.callout(mo.md(f"File **{JSONL_PATH}** not found."), kind="danger"),
    )

    SOURCE_NAME = JSONL_PATH.stem.removesuffix("_units")
    REVIEW_PATH = JSONL_PATH.with_name(f"{SOURCE_NAME}_review.json")
    DRAFTS_PATH = JSONL_PATH.with_name(f"{SOURCE_NAME}_drafts.json")
    EXPORT_PATH = JSONL_PATH.with_name(f"{SOURCE_NAME}_aligned.jsonl")
    return DRAFTS_PATH, EXPORT_PATH, JSONL_PATH, REVIEW_PATH, SOURCE_NAME


@app.cell
def _(JSONL_PATH, json):
    def _load_units(path):
        result = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                (uid, sides), = json.loads(line).items()
                result.append((uid, {"french": sides["fra"], "moore": sides["mos"]}))
        return result

    _units = _load_units(JSONL_PATH)
    original = dict(_units)
    all_ids = [uid for uid, _ in _units]
    mismatched_ids = [uid for uid, sides in original.items() if len(sides["french"]) != len(sides["moore"])]
    return all_ids, mismatched_ids, original


@app.cell
def _(REVIEW_PATH, json, mo, original):
    if REVIEW_PATH.exists():
        _saved = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    else:
        _saved = {}

    _initial = {uid: _saved.get(uid, sides) for uid, sides in original.items()}
    get_review, set_review = mo.state(_initial)
    return get_review, set_review


@app.cell
def _(DRAFTS_PATH, json, mo, original):
    # Raw, un-"Applied" textarea contents, keyed by uid. Saved on every edit
    # (textarea blur / Ctrl+Enter) so closing the notebook mid-edit doesn't
    # lose work that was never committed via "Apply edits".
    if DRAFTS_PATH.exists():
        _saved_drafts = json.loads(DRAFTS_PATH.read_text(encoding="utf-8"))
    else:
        _saved_drafts = {}

    _initial_drafts = {
        uid: _saved_drafts.get(uid, {"french": "\n".join(sides["french"]), "moore": "\n".join(sides["moore"])})
        for uid, sides in original.items()
    }
    get_drafts, set_drafts = mo.state(_initial_drafts)
    return get_drafts, set_drafts


@app.cell
def _(DRAFTS_PATH, REVIEW_PATH, json):
    def save_review(state: dict) -> None:
        REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        REVIEW_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_drafts(state: dict) -> None:
        DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DRAFTS_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    return save_drafts, save_review


@app.cell
def _(get_review, mismatched_ids, mo):
    _resolved = sum(1 for uid in mismatched_ids if len(get_review()[uid]["french"]) == len(get_review()[uid]["moore"]))
    mo.md(f"### Progress: {_resolved} / {len(mismatched_ids)} mismatched units resolved")
    return


@app.cell
def _(all_ids, mo):
    # Every page/enum unit, in book order -- not just the count-mismatched
    # ones, since equal counts don't guarantee correct content (e.g. two
    # sentences swapped). Options must stay stable across reruns (keyed by
    # uid, not a live-computed label) or marimo can't preserve the current
    # selection once an edit changes that unit's sentence counts.
    unit_picker = mo.ui.dropdown(options=all_ids, value=all_ids[0], label="Unit to review")
    unit_picker
    return (unit_picker,)


@app.cell
def _(get_review, mo, unit_picker):
    _sides = get_review()[unit_picker.value]
    _n_fr, _n_mo = len(_sides["french"]), len(_sides["moore"])
    _ok = _n_fr == _n_mo
    _mark = "✅" if _ok else "⚠️"
    mo.callout(
        mo.md(f"{_mark} **{unit_picker.value}** — FR {_n_fr} / MO {_n_mo}"),
        kind="success" if _ok else "warn",
    )
    return


@app.cell
def _(html, mo):
    _STRIPES = ["#eef5ff", "#fff8e6"]

    def colored_preview(items: list[str]) -> object:
        if not items:
            return mo.md("*(empty)*")
        rows_html = "".join(
            f'<div style="background:{_STRIPES[i % 2]}; padding:4px 8px; margin-bottom:2px; '
            f'border-radius:4px; font-family:sans-serif;">'
            f"<b>{i}.</b> {html.escape(sentence)}</div>"
            for i, sentence in enumerate(items)
        )
        return mo.Html(f'<div style="max-height:360px; overflow-y:auto;">{rows_html}</div>')

    return (colored_preview,)


@app.cell
def _(colored_preview, get_drafts, get_review, mo, unit_picker):
    _uid = unit_picker.value
    _sides = get_review()[_uid]
    _draft = get_drafts()[_uid]

    # Free-text editing beats checkbox-per-operation: merge (delete a
    # newline), split (add one), reorder (cut/paste a line), or fix a typo
    # -- all the same primitive, and you can see the whole unit at once
    # instead of one mechanical action at a time. The colored block above
    # each box is a read-only reference view (HTML in a textarea can't be
    # colored) so line boundaries are easy to scan while editing below.
    # Text areas seed from the saved *draft* (raw, possibly un-applied text),
    # not from `_sides`, so reopening the notebook restores exactly what was
    # last typed even if "Apply edits" was never clicked.
    fr_text = mo.ui.text_area(
        value=_draft["french"],
        rows=max(20, len(_sides["french"]) + 3),
        full_width=True,
        label="French — one sentence per line",
    )
    mo_text = mo.ui.text_area(
        value=_draft["moore"],
        rows=max(20, len(_sides["moore"]) + 3),
        full_width=True,
        label="Mooré — one sentence per line",
    )

    mo.vstack(
        [
            mo.md(f"## {_uid}"),
            mo.hstack(
                [
                    mo.vstack([mo.md("### French"), colored_preview(_sides["french"]), fr_text]),
                    mo.vstack([mo.md("### Mooré"), colored_preview(_sides["moore"]), mo_text]),
                ],
                justify="space-between",
                align="start",
                gap=2,
                widths="equal",
            ),
        ]
    )
    return fr_text, mo_text


@app.cell
def _(fr_text, get_drafts, mo_text, save_drafts, set_drafts, unit_picker):
    _uid = unit_picker.value
    _drafts = dict(get_drafts())
    _drafts[_uid] = {"french": fr_text.value, "moore": mo_text.value}
    set_drafts(_drafts)
    save_drafts(_drafts)
    return


@app.cell
def _(mo):
    apply_button = mo.ui.run_button(label="Apply edits")
    reset_button = mo.ui.run_button(label="Reset this unit", kind="danger")
    mo.hstack([apply_button, reset_button], justify="start")
    return apply_button, reset_button


@app.cell
def _(
    apply_button,
    fr_text,
    get_drafts,
    get_review,
    mo,
    mo_text,
    original,
    reset_button,
    save_drafts,
    save_review,
    set_drafts,
    set_review,
    unit_picker,
):
    _uid = unit_picker.value
    _changed = False
    _state = dict(get_review())

    if apply_button.value:
        _fr = [s.strip() for s in fr_text.value.splitlines() if s.strip()]
        _mo = [s.strip() for s in mo_text.value.splitlines() if s.strip()]
        _state[_uid] = {"french": _fr, "moore": _mo}
        _changed = True
    elif reset_button.value:
        _state[_uid] = dict(original[_uid])
        # Reset the draft too, otherwise the textarea would keep showing the
        # stale edited text even though the colored preview above it (which
        # reads get_review directly) has gone back to the original.
        _drafts = dict(get_drafts())
        _drafts[_uid] = {"french": "\n".join(original[_uid]["french"]), "moore": "\n".join(original[_uid]["moore"])}
        set_drafts(_drafts)
        save_drafts(_drafts)
        _changed = True

    if _changed:
        set_review(_state)
        save_review(_state)

    mo.stop(not _changed)
    return


@app.cell
def _(get_review, mismatched_ids, mo):
    export_button = mo.ui.run_button(label="Export resolved pairs to JSONL")

    _still_mismatched = [uid for uid in mismatched_ids if len(get_review()[uid]["french"]) != len(get_review()[uid]["moore"])]

    mo.vstack(
        [
            mo.md(
                "Still mismatched, won't be exported: " + (", ".join(_still_mismatched) if _still_mismatched else "none")
            ),
            export_button,
        ]
    )
    return (export_button,)


@app.cell
def _(EXPORT_PATH, SOURCE_NAME, all_ids, export_button, get_review, json, mo):
    mo.stop(not export_button.value)

    _rows = []
    for _uid in all_ids:
        _sides = get_review()[_uid]
        _fr, _mo = _sides["french"], _sides["moore"]
        if len(_fr) != len(_mo):
            continue
        for _f, _m in zip(_fr, _mo):
            _rows.append({"french": _f, "moore": _m, "source": SOURCE_NAME, "unit": _uid})

    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPORT_PATH, "w", encoding="utf-8") as _fh:
        for _row in _rows:
            _fh.write(json.dumps(_row, ensure_ascii=False) + "\n")

    mo.md(f"Wrote **{len(_rows)}** pairs → `{EXPORT_PATH}`")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
