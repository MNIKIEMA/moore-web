"""Publish Mooré monolingual sentences as ``madoss/moore-web-mono``.

Reads JSONL files of cleaned sentences (``hplt-mono`` output or any file with
the same core fields: ``id``, ``text``, ``source``, ``license``), writes one
parquet folder per source (``data/<source>/train.parquet``) and a dataset card
with a ``default`` config (every source) plus one config per source, so a
user can load only the sources whose license fits.

Adding a source: produce its JSONL, add it to ``SOURCES`` (card text and
license), and run ``publish-mono`` with all the JSONL files.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

MONO_REPO = "madoss/moore-web-mono"

# Card text per source: what it is, where it comes from, its license.
SOURCES = {
    "wikipedia": {
        "title": "Mooré Wikipedia",
        "description": (
            "Articles of the Mooré Wikipedia (mos.wikipedia.org) and the Wikimedia "
            "Incubator (Wp/mos), taken from the HPLT v3 web crawl "
            "([`madoss/mos-latn-hplt`](https://huggingface.co/datasets/madoss/mos-latn-hplt)). "
            "General-knowledge text written by volunteers."
        ),
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Wikipedia contributors; the `url` column links each sentence to its article.",
    },
}

COLUMN_DOCS = [
    ("id", "string", "Content-based id: `hplt-` + 16 hex chars of the SHA-1 of the normalized text"),
    ("text", "string", "One Mooré sentence"),
    ("source", "string", "Source tag (see the table above; also the config name)"),
    ("license", "string", "License of this sentence's source"),
    ("doc_id", "string", "Id of the web document the sentence comes from"),
    ("url", "string", "Page the sentence comes from (attribution)"),
    ("line", "int", "Paragraph number within the page, from 0"),
    ("lang_prob", "float", "GlotLID probability that `text` is Mooré (`mos_Latn`)"),
    ("words", "int", "Word count of `text`"),
]


def read_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def group_by_source(rows: list[dict]) -> dict[str, list[dict]]:
    ids: set[str] = set()
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["source"] not in SOURCES:
            raise ValueError(f"Unknown source {r['source']!r}: add it to SOURCES first")
        if r["license"] != SOURCES[r["source"]]["license"]:
            raise ValueError(
                f"Row {r['id']} has license {r['license']!r}, expected {SOURCES[r['source']]['license']!r}"
            )
        if r["id"] in ids:
            raise ValueError(f"Duplicate id {r['id']}")
        ids.add(r["id"])
        by_source[r["source"]].append(r)
    return dict(sorted(by_source.items()))


def dataset_card(by_source: dict[str, list[dict]]) -> str:
    licenses = {SOURCES[s]["license"] for s in by_source}
    license_yaml = next(iter(licenses)).lower() if len(licenses) == 1 else "other"
    total = sum(len(rows) for rows in by_source.values())
    configs = ["- config_name: default", "  data_files:", "  - split: train", "    path: data/*/*.parquet"]
    for source in by_source:
        configs += [
            f"- config_name: {source}",
            "  data_files:",
            "  - split: train",
            f"    path: data/{source}/*.parquet",
        ]
    header = "\n".join(
        [
            "---",
            f"license: {license_yaml}",
            "language:",
            "- mos",
            "task_categories:",
            "- text-generation",
            "- translation",
            "pretty_name: Mooré monolingual sentences (moore-web)",
            "size_categories:",
            "- 10K<n<100K" if total >= 10_000 else "- 1K<n<10K",
            "tags:",
            "- mooré",
            "- mossi",
            "- burkina-faso",
            "- low-resource",
            "configs:",
            *configs,
            "---",
        ]
    )
    rows_table = "\n".join(
        f"| `{s}` | {SOURCES[s]['title']} | {len(rows):,} | "
        f"{sum(len(r['text']) for r in rows) / 1e6:.2f} M | "
        f"[{SOURCES[s]['license']}]({SOURCES[s]['license_url']}) |"
        for s, rows in by_source.items()
    )
    source_notes = "\n\n".join(
        f"**`{s}`**: {SOURCES[s]['description']} Attribution: {SOURCES[s]['attribution']}" for s in by_source
    )
    columns = "\n".join(f"| `{name}` | {dtype} | {doc} |" for name, dtype, doc in COLUMN_DOCS)
    return f"""{header}

# moore-web-mono

Clean Mooré (Mossi, `mos`) sentences, one per row, for language modeling and
for backtranslation into French. Built by the
[moore-web](https://github.com/MNIKIEMA/moore-web) pipeline. Mooré only: for
French–Mooré pairs see
[`madoss/moore-web-parallel`](https://huggingface.co/datasets/madoss/moore-web-parallel).

```python
from datasets import load_dataset

ds = load_dataset("madoss/moore-web-mono")               # every source
wiki = load_dataset("madoss/moore-web-mono", "wikipedia")  # one source
```

## Sources

| Config / `source` | What | Sentences | Characters | License |
| --- | --- | ---: | ---: | --- |
{rows_table}

{source_notes}

Each source keeps its own license, repeated in the `license` column so it
travels with rows after mixing. Load only the configs whose license fits your
use.

## Fields

| Field | Type | Meaning |
| --- | --- | --- |
{columns}

## How it was built

1. Only sources with clear reuse terms are taken. From the HPLT crawl, only
   Wikipedia pages are kept: most of the crawl is jw.org, whose terms of use
   forbid this reuse.
2. Pages carrying leaked NLLB language tags ("… mos_Latnmos_Latn be be be")
   are dropped whole: they were machine-translated with NLLB.
3. Wikipedia citation markers (`[1]`) are removed, then each paragraph is split
   into sentences.
4. Sentences are kept when GlotLID tags them `mos_Latn` with probability ≥ 0.8,
   contain no letters of another script or phonetic transcriptions, no wiki
   markup, fewer than two lowercase words with tone accents (another spelling
   system or language), at least 4 words and at most 500 characters, and are
   not generation loops (8+ word sentences need at least 55% distinct words).
5. Exact duplicates are removed after normalization (Unicode NFKC, lowercase,
   punctuation removed, spaces collapsed).
6. Sentences that appear in the Mooré references of FLORES+ (dev, devtest) or
   Bouquet (fra–mos, all splits), or on the Mooré side of
   `moore-web-parallel` v1.0.0, are removed, so the data can be used to train
   models evaluated on those benchmarks.

## Limitations

- **Small and single-domain for now**: encyclopedic text only.
- **Quality varies**: Wikipedia and Incubator articles are written by
  volunteers; spelling conventions (ɩ, ʋ, nasal vowels, word breaks) are not
  uniform. Some Incubator articles were machine-translated: pages with leaked
  tags and looping sentences are removed, but machine-translated pages without
  those traces cannot be detected.
- **Language ID errors**: short sentences with names or numbers can be
  misclassified in either direction.
"""


def write_folder(by_source: dict[str, list[dict]], out_dir: Path) -> None:
    from datasets import Dataset

    for source, rows in by_source.items():
        path = out_dir / "data" / source / "train.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [name for name, _, _ in COLUMN_DOCS]
        Dataset.from_list([{c: r.get(c) for c in columns} for r in rows]).to_parquet(str(path))
    (out_dir / "README.md").write_text(dataset_card(by_source), encoding="utf-8")


def push_folder(
    out_dir: Path, repo_id: str = MONO_REPO, private: bool = True, message: str | None = None
) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=out_dir,
        allow_patterns=["README.md", "data/**/*.parquet"],
        delete_patterns=["data/**/*.parquet"],
        commit_message=message or "Update moore-web-mono",
    )
    return commit.oid
