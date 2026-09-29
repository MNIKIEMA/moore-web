"""Clean Mooré monolingual sentences from the HPLT crawl (``madoss/mos-latn-hplt``).

The crawl holds whole web documents already tagged ``mos_Latn`` by HPLT, but
most of them are jw.org publications (terms of use forbid this reuse) and the
rest mix Mooré with English/French page furniture. This keeps only the
documents from the chosen hosts (Wikipedia by default: general-domain text,
CC BY-SA 4.0), drops whole documents with leaked NLLB language tags
("mos_Latnmos_Latn": machine-translated with NLLB), then works sentence by
sentence:

1. strip Wikipedia citation markers (``[1]``, ``[a]``) from each line, then
   split it with ``segment_mo`` (the syntok-based Mooré splitter);
2. keep sentences GlotLID tags ``mos_Latn`` with probability >= ``min_prob``,
   without letters of another script or IPA (glosses like "(Korean: 연등회)"),
   and with fewer than 2 lowercase words carrying tone accents (ó ù ī …: another
   spelling system or language; capitalized foreign names are allowed);
3. drop fragments (< ``min_words`` words), run-ons (> ``max_chars``), and
   generation loops typical of machine translation: sentences of 8+ words where
   fewer than ``min_distinct_ratio`` of the words are distinct;
4. drop exact duplicates after normalization (NFKC, lowercase, punctuation
   removed, spaces collapsed);
5. drop sentences found in the evaluation references (FLORES+ ``mos_Latn``
   dev/devtest, Bouquet fra-mos targets) or in ``exclude_texts`` (e.g. the
   Mooré side of the parallel data).

Each row gets a content-based ``id`` (``hplt-`` + 16 hex chars of the SHA-1 of
the normalized text), unique because dedup uses the same normalization.

The output feeds backtranslation (Mooré → French) in ``mt-training``.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from moore_web.flatten import segment_mo

HPLT_REPO = "madoss/mos-latn-hplt"
SOURCE = "wikipedia"
LICENSE = "CC-BY-SA-4.0"
DEFAULT_HOSTS = ("wikipedia.org", "incubator.wikimedia.org", "incubator.m.wikimedia.org")
MOORE = "mos_Latn"

# "[1]", "[ 3]", "[a]", "[note 2]", "[DM 1]"
_CITATION = re.compile(r"\s*\[\s*(?:\d+|[a-z]|note \d+|[A-Z]{1,4} \d+)\s*\]")
# Leaked NLLB language tags ("… mos_Latnmos_Latn be be be"): the page was
# machine-translated with NLLB, so none of its Mooré is kept.
_MT_TAG = re.compile(r"[a-z]{3}_(?:Latn|Arab|Cyrl|Deva|Ethi|Hans|Hant|Grek|Hang)")
# Raw wiki markup left in the extracted text ("[[File:…|thumb|…]]").
_WIKI_MARKUP = re.compile(r"\[\[|\]\]|\{\{|\}\}|\|thumb|\b(?:File|Fichier|Image):")
# The crawl sometimes lost the "[": "daarã. 3] Burkĩna …".
_ORPHAN_CITATION = re.compile(r"(?:(?<=\s)|^)\d{1,3}\]\s*")
# Tone accents that neither standard Mooré spelling (ã ẽ ĩ õ ũ, ɛ ɩ ʋ) nor French uses.
_TONE_MARKED = set("óòúùíìáǎěǐǒǔāēīōūû")
# IPA letters and modifier letters (ˈ ː) mark phonetic transcriptions, except the
# Mooré letters ɛ ɔ ɩ ʋ and ə (a normal letter in names such as Azerbaijani Rövşən).
_ALLOWED_IPA_BLOCK = set("ɛɔɩʋə")
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")

LangFn = Callable[[list[str]], tuple[list[str], list[float]]]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return _SPACES.sub(" ", _PUNCT.sub(" ", text)).strip()


def has_foreign_script(text: str) -> bool:
    """True if the text has a non-Latin letter (CJK, Greek, Cyrillic, …) or IPA."""
    for char in text:
        code = ord(char)
        if 0x0250 <= code <= 0x02FF and char not in _ALLOWED_IPA_BLOCK:
            return True
        if char.isalpha() and not unicodedata.name(char, "").startswith("LATIN"):
            return True
    return False


def tone_marked_words(text: str) -> int:
    """Lowercase words with tone accents: another spelling system or language.

    Capitalized words are left out: foreign names (Martínez, Bartók) are common in
    good Mooré sentences.
    """
    return sum(1 for w in text.split() if w[:1].islower() and any(c in _TONE_MARKED for c in w))


_WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")


def distinct_word_ratio(text: str) -> float:
    """Distinct words / words, for sentences of 8+ words (1.0 for shorter ones).

    Low values flag generation loops typical of machine translation
    ("b sẽn yaa b sẽn yaa b to wã …").
    """
    words = [w.lower() for w in _WORD.findall(text)]
    return len(set(words)) / len(words) if len(words) >= 8 else 1.0


def sentence_id(text: str) -> str:
    """Content-based id: stable across re-runs, re-splits and re-translations."""
    return "hplt-" + hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:16]


def host_matches(url: str, hosts: Iterable[str]) -> bool:
    netloc = urlparse(url).netloc.lower()
    return any(netloc == h or netloc.endswith("." + h) for h in hosts)


def split_document(doc: dict) -> list[dict]:
    """One row per sentence: the document's id and URL, line index, sentence."""
    rows = []
    for line_index, line in enumerate(doc["text"].split("\n")):
        # Before splitting: "ye.[1] A" would not be split after "ye.".
        line = _ORPHAN_CITATION.sub("", _CITATION.sub("", line)).strip()
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
    mt_tagged_documents: int = 0
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
    min_distinct_ratio: float = 0.55,
    exclude_texts: Iterable[str] = (),
    source: str = SOURCE,
    license: str = LICENSE,
) -> tuple[list[dict], Stats]:
    stats = Stats()
    docs = [d for d in documents if host_matches(d["u"], hosts)]
    tagged = [d for d in docs if _MT_TAG.search(d["text"])]
    stats.mt_tagged_documents = len(tagged)
    docs = [d for d in docs if not _MT_TAG.search(d["text"])]
    stats.documents = len(docs)
    rows = stats.record("sentences", [r for d in docs for r in split_document(d)])

    langs, probs = lang_fn([r["text"] for r in rows])
    for r, lang, prob in zip(rows, langs, probs, strict=True):
        r["lang"], r["lang_prob"] = lang, round(float(prob), 4)
    rows = stats.record(f"GlotLID {MOORE}", [r for r in rows if r["lang"] == MOORE])
    rows = stats.record(f"prob >= {min_prob}", [r for r in rows if r["lang_prob"] >= min_prob])
    rows = stats.record("no foreign script or IPA", [r for r in rows if not has_foreign_script(r["text"])])
    rows = stats.record("no wiki markup", [r for r in rows if not _WIKI_MARKUP.search(r["text"])])
    rows = stats.record(
        "< 2 tone-marked lowercase words", [r for r in rows if tone_marked_words(r["text"]) < 2]
    )
    rows = stats.record(
        f"{min_words}+ words, <= {max_chars} chars",
        [r for r in rows if len(r["text"].split()) >= min_words and len(r["text"]) <= max_chars],
    )

    rows = stats.record(
        f"distinct words >= {min_distinct_ratio:.0%}",
        [r for r in rows if distinct_word_ratio(r["text"]) >= min_distinct_ratio],
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
    # Dedup ran on the same normalized text, so ids are unique.
    rows = [
        {"id": sentence_id(r["text"]), "text": r["text"], "source": source, "license": license}
        | {k: v for k, v in r.items() if k not in ("lang", "text")}
        | {"words": len(r["text"].split())}
        for r in rows
    ]
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
