"""Clean Mooré monolingual sentences from the HPLT crawl (``madoss/mos-latn-hplt``).

The crawl holds whole web documents already tagged ``mos_Latn`` by HPLT, but
most of them are jw.org publications (terms of use forbid this reuse) and the
rest mix Mooré with English/French page furniture. This keeps only the
documents from the chosen hosts (Wikipedia by default: general-domain text,
CC BY-SA 4.0), then works sentence by sentence:

1. strip Wikipedia citation markers (``[1]``, ``[a]``) from each line, then
   split it with ``segment_mo`` (the syntok-based Mooré splitter);
2. keep sentences GlotLID tags ``mos_Latn`` with probability >= ``min_prob``;
3. drop fragments (< ``min_words`` words) and run-ons (> ``max_chars``);
4. drop exact duplicates after normalization (NFKC, lowercase, punctuation
   removed, spaces collapsed);
5. drop sentences found in the evaluation references (FLORES+ ``mos_Latn``
   dev/devtest, Bouquet fra-mos targets) or in ``exclude_texts`` (e.g. the
   Mooré side of the parallel data).

The output feeds backtranslation (Mooré → French) in ``mt-training``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from moore_web.flatten import segment_mo

HPLT_REPO = "madoss/mos-latn-hplt"
DEFAULT_HOSTS = ("wikipedia.org", "incubator.wikimedia.org", "incubator.m.wikimedia.org")
MOORE = "mos_Latn"

_CITATION = re.compile(r"\s*\[(?:\d+|[a-z]|note \d+)\]")
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")

LangFn = Callable[[list[str]], tuple[list[str], list[float]]]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return _SPACES.sub(" ", _PUNCT.sub(" ", text)).strip()


def host_matches(url: str, hosts: Iterable[str]) -> bool:
    netloc = urlparse(url).netloc.lower()
    return any(netloc == h or netloc.endswith("." + h) for h in hosts)


def split_document(doc: dict) -> list[dict]:
    """One row per sentence: the document's id and URL, line index, sentence."""
    rows = []
    for line_index, line in enumerate(doc["text"].split("\n")):
        # Before splitting: "ye.[1] A" would not be split after "ye.".
        line = _CITATION.sub("", line).strip()
        if not line:
            continue
        for sentence in segment_mo(line):
            sentence = sentence.strip()
            if sentence:
                rows.append({"doc_id": doc["id"], "url": doc["u"], "line": line_index, "text": sentence})
    return rows


@dataclass
class Stats:
    documents: int = 0
    steps: dict[str, int] = field(default_factory=dict)

    def record(self, step: str, rows: list[dict]) -> list[dict]:
        self.steps[step] = len(rows)
        return rows


def clean_sentences(
    documents: list[dict],
    lang_fn: LangFn,
    *,
    hosts: Iterable[str] = DEFAULT_HOSTS,
    min_prob: float = 0.8,
    min_words: int = 4,
    max_chars: int = 500,
    exclude_texts: Iterable[str] = (),
) -> tuple[list[dict], Stats]:
    stats = Stats()
    docs = [d for d in documents if host_matches(d["u"], hosts)]
    stats.documents = len(docs)
    rows = stats.record("sentences", [r for d in docs for r in split_document(d)])

    langs, probs = lang_fn([r["text"] for r in rows])
    for r, lang, prob in zip(rows, langs, probs, strict=True):
        r["lang"], r["lang_prob"] = lang, round(float(prob), 4)
    rows = stats.record(f"GlotLID {MOORE}", [r for r in rows if r["lang"] == MOORE])
    rows = stats.record(f"prob >= {min_prob}", [r for r in rows if r["lang_prob"] >= min_prob])
    rows = stats.record(
        f"{min_words}+ words, <= {max_chars} chars",
        [r for r in rows if len(r["text"].split()) >= min_words and len(r["text"]) <= max_chars],
    )

    seen: set[str] = set()
    unique = []
    for r in rows:
        key = normalize(r["text"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    rows = stats.record("deduplicated", unique)

    excluded = {normalize(t) for t in exclude_texts}
    rows = stats.record("not in excluded texts", [r for r in rows if normalize(r["text"]) not in excluded])
    for r in rows:
        r["words"] = len(r["text"].split())
        del r["lang"]
    return rows, stats


def evaluation_references() -> list[str]:
    """Mooré references of FLORES+ (dev, devtest) and Bouquet fra-mos (all levels/splits)."""
    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    texts: list[str] = []
    for split in ("dev", "devtest"):
        texts += load_dataset("openlanguagedata/flores_plus", MOORE, split=split)["text"]
    for level in ("sentence_level", "paragraph_level"):
        for split in ("dev", "test"):
            path = hf_hub_download(
                "facebook/bouquet",
                f"benchmark_data/{level}/{split}/fra_Latn-mos_Latn.parquet",
                repo_type="dataset",
            )
            texts += pd.read_parquet(path)["tgt_text"].tolist()
    return texts


def load_hplt_documents(repo_id: str = HPLT_REPO) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(repo_id, split="train")
    return [{"id": r["id"], "u": r["u"], "text": r["text"]} for r in ds]


def glotlid_lang_fn() -> LangFn:
    from moore_web.glotlid import load_model, predict

    model = load_model()

    def lang_fn(texts: list[str]) -> tuple[list[str], list[float]]:
        langs, probs = predict(model, texts)
        return langs.tolist(), probs.tolist()

    return lang_fn


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
