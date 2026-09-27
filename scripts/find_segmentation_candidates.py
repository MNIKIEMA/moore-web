"""List review-app lines that still hold several sentences and may need splitting.

Run from the repository root:
    uv run python scripts/find_segmentation_candidates.py mos-contes-volume-5

With --draft REVIEWER, EASY and QUOTE lines (plus DIALOGUE with --all) are split at
their sentence ends and saved as that reviewer's draft for the unit. The accepted
review is not touched: the split only replaces it once the draft is marked reviewed
in the app. CHECK lines are left as they are, and units that already have a draft
from that reviewer are skipped.
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

from moore_web import review_store

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data/review/reviews.sqlite3"

# A sentence ends at . ! ? … (optionally followed by a closing quote) when a capital comes next.
SENTENCE_END = re.compile(r'[.!?…]+[»"\s]*(?=\s+[«"\-–—]?\s*[A-ZÀ-ÖØ-Þ])')


def count_sentences(text: str) -> int:
    return len(SENTENCE_END.findall(text)) + 1


def split_sentences(text: str) -> list[str]:
    pieces, start = [], 0
    for match in SENTENCE_END.finditer(text):
        pieces.append(text[start : match.end()].strip())
        start = match.end()
    pieces.append(text[start:].strip())
    return [piece for piece in pieces if piece]


def split_side(lines: list[str], rejected: set[int], to_split: set[int]) -> tuple[list[str], list[int]]:
    """Split the lines at `to_split` (0-based), re-pointing rejected indices at the new list."""
    new_lines: list[str] = []
    new_rejected: list[int] = []
    for index, line in enumerate(lines):
        if index in rejected:
            new_rejected.append(len(new_lines))
        new_lines.extend(split_sentences(line) if index in to_split else [line])
    return new_lines, new_rejected


def has_quote(text: str) -> bool:
    return "«" in text or '"' in text


def classify(fra: str, mos: str, max_chars: int) -> str | None:
    n_fra, n_mos = count_sentences(fra), count_sentences(mos)
    long = max(len(fra), len(mos)) > max_chars
    if n_fra < 2 and n_mos < 2 and not long:
        return None
    if n_fra != n_mos and (n_fra >= 2 and n_mos >= 2 or long):
        return "check"
    if n_fra < 2 or n_mos < 2:
        return None
    if not has_quote(fra):
        return "easy"
    if n_fra >= 3 or long:
        return "quote"
    return "short-dialogue"


LABELS = {
    "easy": "EASY     narrative, same sentence count",
    "quote": "QUOTE    long quoted speech",
    "check": "CHECK    sentence counts differ",
    "short-dialogue": "DIALOGUE two-sentence quote",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="review source, e.g. mos-contes-volume-5")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    # NLLB fine-tuning uses max_length=256 tokens; Mooré runs ~0.40 tokens/char, so
    # 350 chars is ~140 tokens and leaves headroom for the longer side of a pair.
    parser.add_argument("--max-chars", type=int, default=350)
    parser.add_argument("--all", action="store_true", help="also list short two-sentence dialogue lines")
    parser.add_argument("--width", type=int, default=160, help="truncate texts (0 = full)")
    parser.add_argument(
        "--draft", metavar="REVIEWER", help="save the proposed splits as this reviewer's drafts"
    )
    args = parser.parse_args()

    db = sqlite3.connect(args.db)
    rows = db.execute(
        """SELECT u.id, u.unit_uid, r.fra, r.mos, r.rejected_fra, r.rejected_mos, r.version
        FROM units u JOIN reviews r ON r.unit_id = u.id
        WHERE u.source = ? ORDER BY u.position, u.id""",
        (args.source,),
    ).fetchall()
    if not rows:
        raise SystemExit(f"No units for source {args.source!r} in {args.db}")

    def clip(text: str) -> str:
        return text if not args.width or len(text) <= args.width else text[: args.width] + "…"

    totals = {kind: 0 for kind in LABELS}
    n_pairs = 0
    drafted, skipped = [], []
    for unit_id, unit_uid, fra_json, mos_json, rej_fra, rej_mos, version in rows:
        fra, mos = json.loads(fra_json), json.loads(mos_json)
        skip_fra, skip_mos = set(json.loads(rej_fra)), set(json.loads(rej_mos))
        # Keep editor line numbers (1-based, rejected lines included) so they match the app.
        kept_fra = [(i + 1, t) for i, t in enumerate(fra) if i not in skip_fra]
        kept_mos = [(i + 1, t) for i, t in enumerate(mos) if i not in skip_mos]
        hits = []
        for (fra_line, fra_text), (mos_line, mos_text) in zip(kept_fra, kept_mos):
            n_pairs += 1
            kind = classify(fra_text, mos_text, args.max_chars)
            if kind is None or (kind == "short-dialogue" and not args.all):
                continue
            totals[kind] += 1
            hits.append((kind, fra_line, mos_line, fra_text, mos_text))
        if not hits:
            continue
        print(f"\n=== {unit_uid}  (db id {unit_id}) — {len(hits)} line(s)")
        for kind, fra_line, mos_line, fra_text, mos_text in hits:
            line = f"line {fra_line}" if fra_line == mos_line else f"fr line {fra_line} / mos line {mos_line}"
            print(
                f"  [{LABELS[kind]}] {line}: "
                f"fr {count_sentences(fra_text)}s/{len(fra_text)}ch, "
                f"mos {count_sentences(mos_text)}s/{len(mos_text)}ch"
            )
            print(f"    FR : {clip(fra_text)}")
            print(f"    MOS: {clip(mos_text)}")

        if args.draft:
            splits = [(f - 1, m - 1) for kind, f, m, *_ in hits if kind != "check"]
            if not splits:
                continue
            if review_store.get_draft(args.db, unit_id, args.draft):
                skipped.append(unit_uid)
                continue
            new_fra, new_rej_fra = split_side(fra, skip_fra, {f for f, _ in splits})
            new_mos, new_rej_mos = split_side(mos, skip_mos, {m for _, m in splits})
            assert len(new_fra) - len(new_rej_fra) == len(new_mos) - len(new_rej_mos), unit_uid
            review_store.save_draft(
                args.db,
                unit_id,
                args.draft,
                "\n".join(new_fra),
                "\n".join(new_mos),
                version,
                new_rej_fra,
                new_rej_mos,
            )
            drafted.append(f"{unit_uid} ({len(splits)} line(s) split)")

    print(f"\n{args.source}: {len(rows)} units, {n_pairs} kept pairs")
    for kind, label in LABELS.items():
        if kind != "short-dialogue" or args.all:
            print(f"  {label}: {totals[kind]}")
    if args.draft:
        print(f"\nDrafts saved for {args.draft}: {len(drafted)}")
        for name in drafted:
            print(f"  {name}")
        if skipped:
            print(f"Skipped (draft already exists): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
