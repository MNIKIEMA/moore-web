"""Review every bilingual unit in one Shiny page.

Run from the repository root: uv run shiny run apps/review_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from htmltools import Tag
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from moore_web import review_store

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = Path(os.environ.get("REVIEW_INPUT_DIR", ROOT / "data/review"))
DB_PATH = Path(os.environ.get("REVIEW_DB_PATH", INPUT_DIR / "reviews.sqlite3"))

CSS = """
body { background: #f7f8fa; }
.review-shell { max-width: 1500px; margin: 0 auto; padding: 22px 16px 60px; }
.review-toolbar { display: flex; align-items: end; gap: 18px; flex-wrap: wrap; margin: 18px 0; }
.review-toolbar .form-group { min-width: 240px; margin-bottom: 0; }
.review-unit { background: white; border: 1px solid #d8dee7; border-radius: 8px;
  margin: 10px 0; padding: 0 14px; box-shadow: 0 1px 3px #0000000d; }
.review-unit summary { display: flex; align-items: center; gap: 10px; cursor: pointer;
  padding: 13px 0; font-weight: 600; }
.review-unit summary::marker { color: #64748b; }
.review-unit .counts { color: #475569; font-weight: 400; margin-left: auto; }
.review-unit .badge { font-size: 0.78rem; }
.unit-body { padding-bottom: 14px; }
.parallel { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }
.side { min-width: 0; border-radius: 6px; padding: 10px 12px; }
.side.fr { background: #eef5ff; }
.side.mo { background: #fff8e6; }
.side h4 { font-size: 0.95rem; margin: 0 0 8px; }
.side ol { margin: 0; padding-left: 24px; }
.side li { margin: 4px 0; padding: 3px 8px; border-left: 5px solid var(--line-bd, #94a3b8);
  background: var(--line-bg, #fff); border-radius: 4px; white-space: pre-wrap; overflow-wrap: anywhere; }
.side li::marker { font-weight: 700; color: var(--line-bd, #64748b); }
.side li.pair-0 { --line-bd: #2563eb; --line-bg: #dbeafe; }
.side li.pair-1 { --line-bd: #16a34a; --line-bg: #dcfce7; }
.side li.pair-2 { --line-bd: #d97706; --line-bg: #fef3c7; }
.side li.pair-3 { --line-bd: #9333ea; --line-bg: #f3e8ff; }
.side li.pair-4 { --line-bd: #0d9488; --line-bg: #ccfbf1; }
.side li.pair-5 { --line-bd: #db2777; --line-bg: #fce7f3; }
.side li.unpaired { --line-bd: #dc2626; --line-bg: #fee2e2; border-left-style: dashed; }
.legend { color: #475569; font-size: 0.85rem; margin: 0 0 8px; }
.legend .unpaired-swatch { display: inline-block; width: 12px; height: 12px; margin: 0 4px -1px 0;
  background: #fee2e2; border: 1px dashed #dc2626; border-radius: 2px; }
.unit-actions { display: flex; gap: 10px; align-items: center; margin: 12px 0 2px; }
.modal-dialog { max-width: min(1200px, 96vw); }
.editor-text .form-control { font-family: inherit; line-height: 1.5; }
@media (max-width: 700px) { .parallel { grid-template-columns: 1fr; } }
"""


PAIR_COLORS = 6


def _line_class(index: int, other_count: int) -> str:
    """Same colour for line N on both sides; red when the other side has no line N."""
    return "unpaired" if index >= other_count else f"pair-{index % PAIR_COLORS}"


def _side(title: str, sentences: list[str], class_name: str, other_count: int) -> Tag:
    return ui.div(
        ui.tags.h4(title),
        ui.tags.ol(*(ui.tags.li(s, class_=_line_class(i, other_count)) for i, s in enumerate(sentences))),
        class_=f"side {class_name}",
    )


def _parallel(unit: dict) -> Tag:
    return ui.div(
        ui.p(
            "Line N has the same colour on both sides. ",
            ui.span(class_="unpaired-swatch"),
            "Red dashed = no counterpart on the other side.",
            class_="legend",
        ),
        ui.div(
            _side("French", unit["fra"], "fr", len(unit["mos"])),
            _side("Mooré", unit["mos"], "mo", len(unit["fra"])),
            class_="parallel",
        ),
    )


def _unit_card(unit: dict) -> object:
    counts = f"FR {len(unit['fra'])} / MO {len(unit['mos'])}"
    if unit["reviewed"]:
        status = ui.span("Reviewed", class_="badge bg-success")
    elif len(unit["fra"]) != len(unit["mos"]):
        status = ui.span("Count mismatch", class_="badge bg-warning text-dark")
    else:
        status = ui.span("Pending", class_="badge bg-secondary")
    return ui.tags.details(
        ui.tags.summary(
            ui.span(f"{unit['source']} · {unit['unit_uid']}"),
            status,
            ui.span(counts, class_="counts"),
        ),
        ui.div(
            _parallel(unit),
            ui.div(
                ui.tags.button(
                    "Edit / review",
                    type="button",
                    class_="btn btn-sm btn-outline-primary",
                    onclick=f"Shiny.setInputValue('edit_unit', {unit['id']}, {{priority: 'event'}})",
                ),
                ui.span(
                    f"Last reviewed by {unit['reviewed_by']}" if unit["reviewed_by"] else "",
                    class_="text-muted",
                ),
                class_="unit-actions",
            ),
            class_="unit-body",
        ),
        class_="review-unit",
        open=True,
    )


def _editor_modal(unit: dict, draft: dict | None) -> object:
    fra_text = draft["fra_text"] if draft else "\n".join(unit["fra"])
    mos_text = draft["mos_text"] if draft else "\n".join(unit["mos"])
    stale = draft is not None and draft["base_version"] != unit["version"]
    conflict_message = (
        ui.div(
            ui.p(
                "The accepted version changed after this draft began. Compare it below, then choose whether to use the latest version as your base."
            ),
            ui.input_action_button("rebase", "Use latest version as base", class_="btn-warning"),
            class_="alert alert-warning",
        )
        if stale
        else None
    )
    return ui.modal(
        conflict_message,
        ui.p(
            "One sentence per line. Remove a line break to merge; add one to split; cut and paste to reorder."
        ),
        ui.div(
            ui.div(
                ui.input_text_area("edit_fra", "French", value=fra_text, rows=18, width="100%"),
                class_="editor-text side fr",
            ),
            ui.div(
                ui.input_text_area("edit_mos", "Mooré", value=mos_text, rows=18, width="100%"),
                class_="editor-text side mo",
            ),
            class_="parallel",
        ),
        ui.output_text("editor_counts"),
        ui.tags.details(
            ui.tags.summary("Current accepted text for comparison"),
            _parallel(unit),
        ),
        title=f"{unit['source']} · {unit['unit_uid']}",
        footer=ui.div(
            ui.input_action_button("restore_source", "Restore source text", class_="btn-outline-secondary"),
            ui.input_action_button("save_draft", "Save draft", class_="btn-secondary"),
            ui.input_action_button("accept_review", "Mark reviewed", class_="btn-primary"),
            ui.modal_button("Close"),
            class_="d-flex gap-2",
        ),
        easy_close=False,
        size="l",
    )


def make_app(db_path: Path = DB_PATH, input_dir: Path = INPUT_DIR) -> App:
    review_store.initialize(db_path)
    review_store.import_units(db_path, list(input_dir.glob("*_units.jsonl")))
    source_choices = {"": "All sources", **{source: source for source in review_store.list_sources(db_path)}}

    app_ui = ui.page_fluid(
        ui.tags.style(CSS),
        ui.div(
            ui.h1("Bilingual unit review"),
            ui.p("Browse imported units in manageable pages and open any unit to review its sentences."),
            ui.div(
                ui.input_text("reviewer", "Reviewer name", placeholder="Enter your name before editing"),
                ui.input_select(
                    "status_filter",
                    "Status",
                    {
                        "all": "All units",
                        "pending": "Pending",
                        "mismatched": "Count mismatch",
                        "reviewed": "Reviewed",
                    },
                ),
                ui.input_select("source_filter", "Source", source_choices),
                ui.input_select("page_size", "Units per page", {"25": "25", "50": "50", "100": "100"}),
                ui.input_action_button("refresh", "Refresh data"),
                ui.tags.button(
                    "Collapse all",
                    type="button",
                    class_="btn btn-outline-secondary",
                    onclick="document.querySelectorAll('.review-unit').forEach(unit => unit.open = false)",
                ),
                ui.tags.button(
                    "Expand all",
                    type="button",
                    class_="btn btn-outline-secondary",
                    onclick="document.querySelectorAll('.review-unit').forEach(unit => unit.open = true)",
                ),
                ui.download_button("download_pairs", "Download aligned pairs JSONL"),
                ui.download_button("download_reviewed", "Download reviewed only"),
                class_="review-toolbar",
            ),
            ui.output_text("progress"),
            ui.div(
                ui.input_action_button("previous_page", "Previous", class_="btn-outline-primary"),
                ui.output_text("page_info", inline=True),
                ui.input_action_button("next_page", "Next", class_="btn-outline-primary"),
                class_="d-flex align-items-center gap-3 my-3",
            ),
            ui.output_ui("all_units"),
            class_="review-shell",
        ),
        title="Bilingual unit review",
    )

    def server(input: Inputs, output: Outputs, session: Session) -> None:
        current_id = reactive.value(None)
        current_reviewer = reactive.value("")
        base_version = reactive.value(0)
        refresh_count = reactive.value(0)
        current_page = reactive.value(1)

        @reactive.calc
        def filtered_total() -> int:
            refresh_count()
            return review_store.count_units(
                db_path,
                status=input.status_filter(),
                source=input.source_filter() or None,
            )

        @reactive.calc
        def page_count() -> int:
            page_size = int(input.page_size())
            return max(1, (filtered_total() + page_size - 1) // page_size)

        @reactive.effect
        def _clamp_page() -> None:
            if current_page() > page_count():
                current_page.set(page_count())

        @render.text
        def progress() -> str:
            refresh_count()
            summary = review_store.review_summary(db_path)
            return (
                f"{summary['total']} units · {summary['reviewed']} reviewed · "
                f"{summary['mismatched']} pending with different sentence counts"
            )

        @render.text
        def page_info() -> str:
            total = filtered_total()
            if not total:
                return "No matching units"
            page_size = int(input.page_size())
            first = (current_page() - 1) * page_size + 1
            last = min(current_page() * page_size, total)
            return f"{first}–{last} of {total} · page {current_page()} of {page_count()}"

        @render.ui
        def all_units() -> object:
            refresh_count()
            page_size = int(input.page_size())
            units = review_store.list_units(
                db_path,
                limit=page_size,
                offset=(current_page() - 1) * page_size,
                status=input.status_filter(),
                source=input.source_filter() or None,
            )
            if not units:
                return ui.p("No units match the current filters.")
            return ui.div(*(_unit_card(unit) for unit in units))

        @reactive.effect
        @reactive.event(input.status_filter, input.source_filter, input.page_size)
        def _reset_page() -> None:
            current_page.set(1)

        @reactive.effect
        @reactive.event(input.previous_page)
        def _previous_page() -> None:
            current_page.set(max(1, current_page() - 1))

        @reactive.effect
        @reactive.event(input.next_page)
        def _next_page() -> None:
            current_page.set(min(page_count(), current_page() + 1))

        @reactive.effect
        @reactive.event(input.refresh)
        def _refresh() -> None:
            refresh_count.set(refresh_count() + 1)

        @reactive.effect
        @reactive.event(input.edit_unit)
        def _open_editor() -> None:
            reviewer = input.reviewer().strip()
            if not reviewer:
                ui.notification_show("Enter a reviewer name before editing.", type="warning")
                return
            unit_id = int(input.edit_unit())
            unit = review_store.get_unit(db_path, unit_id)
            draft = review_store.get_draft(db_path, unit_id, reviewer)
            current_id.set(unit_id)
            current_reviewer.set(reviewer)
            base_version.set(draft["base_version"] if draft else unit["version"])
            ui.modal_show(_editor_modal(unit, draft))

        @render.text
        def editor_counts() -> str:
            fra = review_store.split_lines(input.edit_fra())
            mos = review_store.split_lines(input.edit_mos())
            return f"Current edit: FR {len(fra)} / MO {len(mos)}"

        @reactive.effect
        @reactive.event(input.restore_source)
        def _restore_source() -> None:
            if current_id() is None:
                return
            unit = review_store.get_unit(db_path, current_id())
            ui.update_text_area("edit_fra", value="\n".join(unit["original_fra"]))
            ui.update_text_area("edit_mos", value="\n".join(unit["original_mos"]))
            ui.notification_show(
                "Source text loaded into the editor. Save the draft or mark reviewed to keep it."
            )

        @reactive.effect
        @reactive.event(input.save_draft)
        def _save_draft() -> None:
            if current_id() is None:
                return
            review_store.save_draft(
                db_path, current_id(), current_reviewer(), input.edit_fra(), input.edit_mos(), base_version()
            )
            ui.notification_show("Draft saved.", type="message")

        @reactive.effect
        @reactive.event(input.rebase)
        def _rebase() -> None:
            if current_id() is None:
                return
            latest = review_store.get_unit(db_path, current_id())
            base_version.set(latest["version"])
            review_store.save_draft(
                db_path,
                current_id(),
                current_reviewer(),
                input.edit_fra(),
                input.edit_mos(),
                latest["version"],
            )
            ui.notification_show(
                "Draft now uses the latest version as its base. Review your changes before saving.",
                type="message",
            )

        @reactive.effect
        @reactive.event(input.accept_review)
        def _accept() -> None:
            if current_id() is None:
                return
            fra_text, mos_text = input.edit_fra(), input.edit_mos()
            review_store.save_draft(
                db_path, current_id(), current_reviewer(), fra_text, mos_text, base_version()
            )
            try:
                review_store.accept_review(
                    db_path, current_id(), current_reviewer(), fra_text, mos_text, base_version()
                )
            except (ValueError, review_store.ReviewConflict) as exc:
                ui.notification_show(str(exc), type="error", duration=10)
                return
            ui.modal_remove()
            current_id.set(None)
            refresh_count.set(refresh_count() + 1)
            ui.notification_show("Unit marked reviewed.", type="message")

        @render.download_button(filename="aligned.jsonl", media_type="application/x-ndjson")
        def download_pairs():
            for pair in review_store.iter_pairs(db_path):
                yield json.dumps(pair, ensure_ascii=False) + "\n"

        @render.download_button(filename="reviewed_aligned.jsonl", media_type="application/x-ndjson")
        def download_reviewed():
            for pair in review_store.iter_pairs(db_path, reviewed_only=True):
                yield json.dumps(pair, ensure_ascii=False) + "\n"

    return App(app_ui, server)


app = make_app()
