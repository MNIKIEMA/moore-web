"""Scan French–Mooré pairs for Unicode problems: decomposed accents, look-alike letters.

Reports, per side, the rows not in NFC form, the known look-alikes that
``moore_web.orthography.normalize_moore`` fixes (Greek ι for ɩ, ɡ for g, …),
and any other unusual letter (neither Latin with an ASCII base nor a Mooré
letter), with counts per source and an example. Each is shown as the data is
and after the build's normalization, so this checks both a source file and a
published release.

Run from the repository root:
    uv run python scripts/check_orthography.py                      # moore-web-parallel v1.0.0
    uv run python scripts/check_orthography.py --revision v1.1.0 --strict
    uv run python scripts/check_orthography.py --jsonl data/reviewed/*.jsonl

--strict exits with status 1 if, as the data is, any Mooré row is not NFC or
contains a known look-alike (i.e. the data still needs the build's fixes).
"""

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from moore_web.orthography import (
    ALLOWED_MARKS,
    FRENCH_LETTERS,
    FRENCH_LOOKALIKES,
    MOORE_LETTERS,
    MOORE_LOOKALIKES,
    normalize_french,
    normalize_moore,
    unusual_letters,
)


def load_rows(args: argparse.Namespace) -> list[dict]:
    if args.jsonl:
        rows = []
        for path in args.jsonl:
            with Path(path).open(encoding="utf-8") as f:
                rows += [json.loads(line) for line in f if line.strip()]
        return rows
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config, revision=args.revision)
    return [r for split in ds.values() for r in split]


def report(
    rows: list[dict],
    column: str,
    allowed: set[str],
    lookalikes: dict,
    normalize,
    max_examples: int,
    allowed_marks: set[str] | None = ALLOWED_MARKS,
) -> dict:
    not_nfc: Counter = Counter()
    letters: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, str] = {}
    after_letters: Counter = Counter()
    after_not_nfc = 0
    for r in rows:
        text = r.get(column) or ""
        source = r.get("source", "?")
        if unicodedata.normalize("NFC", text) != text:
            not_nfc[source] += 1
        for char in unusual_letters(text, allowed) | {c for c in text if c == "͂"}:
            letters[char][source] += 1
            examples.setdefault(char, text)
        fixed = normalize(text)
        after_not_nfc += unicodedata.normalize("NFC", fixed) != fixed
        after_letters.update(unusual_letters(fixed, allowed, allowed_marks))

    print(f"\n== {column}: {len(rows):,} rows")
    print(
        f"  not NFC: {sum(not_nfc.values()):,} {dict(not_nfc.most_common(4))}  -> after normalization: {after_not_nfc}"
    )
    if not letters:
        print("  unusual letters: none")
    for char, sources in sorted(letters.items(), key=lambda x: -sum(x[1].values()))[:max_examples]:
        text = examples[char]
        i = text.index(char)
        kind = f"look-alike -> {lookalikes[char]!r}" if char in lookalikes else "review"
        print(
            f"  {char!r} U+{ord(char):04X} {unicodedata.name(char, '?')[:30]:30} {sum(sources.values()):>5} rows "
            f"{dict(sources.most_common(2))}  [{kind}]  …{text[max(0, i - 15) : i + 10]!r}"
        )
    remaining = {c: n for c, n in after_letters.items() if c in lookalikes}
    print(f"  known look-alikes left after normalization: {remaining or 'none'}")
    return {
        "not_nfc": sum(not_nfc.values()),
        "lookalikes": sum(n for c, s in letters.items() if c in lookalikes for n in s.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="madoss/moore-web-parallel")
    parser.add_argument("--config", default="mos-fra")
    parser.add_argument("--revision", default="v1.0.0")
    parser.add_argument("--jsonl", nargs="*", help="Local JSONL files with french/moore columns instead")
    parser.add_argument("--max-letters", type=int, default=20, help="Unusual letters to list per side")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if Mooré rows still need the fixes")
    args = parser.parse_args()

    rows = load_rows(args)
    source = ", ".join(args.jsonl) if args.jsonl else f"{args.dataset} {args.config} @ {args.revision}"
    print(f"Checking {source}")
    mos = report(rows, "moore", MOORE_LETTERS, MOORE_LOOKALIKES, normalize_moore, args.max_letters)
    report(rows, "french", FRENCH_LETTERS, FRENCH_LOOKALIKES, normalize_french, args.max_letters, None)
    if args.strict and (mos["not_nfc"] or mos["lookalikes"]):
        print(f"\nFAIL: {mos['not_nfc']} Mooré rows not NFC, {mos['lookalikes']} with look-alikes")
        sys.exit(1)


if __name__ == "__main__":
    main()
