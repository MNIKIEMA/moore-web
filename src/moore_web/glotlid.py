"""Language identification using the GlotLID fasttext model from HuggingFace.

Usage
-----
    # Annotate a HF dataset and push to Hub
    uv run python -m moore_web.glotlid --source-repo madoss/nllb-mos-raw --hub-repo madoss/nllb-mos-lid

    # Custom source/target columns
    uv run python -m moore_web.glotlid --source-repo madoss/nllb-mos-raw --hub-repo madoss/nllb-mos-lid \\
        --source-col eng_Latn --target-col mos_Latn
"""

from __future__ import annotations

import argparse
from dotenv import load_dotenv
import fasttext
import pandas as pd
from huggingface_hub import hf_hub_download

load_dotenv()

REPO_ID = "cis-lmu/glotlid"
FILENAME = "model.bin"


def load_model(repo_id: str = REPO_ID) -> fasttext.FastText._FastText:
    """Download and load the GlotLID fasttext model from HuggingFace Hub."""
    model_path = hf_hub_download(repo_id=repo_id, filename=FILENAME)
    return fasttext.load_model(model_path)


def predict(model: fasttext.FastText._FastText, texts: list[str], k: int = 1) -> tuple[pd.Series, pd.Series]:
    """Return (predicted_language, predicted_probability) series for each text."""
    # fasttext expects no newlines
    cleaned = [t.replace("\n", " ") for t in texts]
    labels, probs = model.predict(cleaned, k=k)
    # labels are like ['__label__mos_Latn'], strip the prefix
    langs = pd.Series([line[0].replace("__label__", "") for line in labels], name="predicted_language")
    scores = pd.Series([round(float(p[0]), 4) for p in probs], name="predicted_probability")
    return langs, scores


def detect_for_texts(texts: list[str], model: fasttext.FastText._FastText | None = None) -> pd.DataFrame:
    """Run lang ID on a flat list of strings. Returns a DataFrame with columns:
    text, predicted_language, predicted_probability.
    """
    if model is None:
        model = load_model()
    langs, probs = predict(model, texts)
    return pd.DataFrame({"text": texts, "predicted_language": langs, "predicted_probability": probs})


def annotate_dataset(
    dataset,
    model: fasttext.FastText._FastText | None = None,
    source_col: str = "eng_Latn",
    target_col: str = "mos_Latn",
    batch_size: int = 1000,
):
    """Add GlotLID predictions to a HuggingFace Dataset.

    Adds four new columns derived from the input column names:
      ``{source_col}_glotlid_lang``, ``{source_col}_glotlid_prob``,
      ``{target_col}_glotlid_lang``, ``{target_col}_glotlid_prob``.
    """
    if model is None:
        model = load_model()

    src_lang_col = f"{source_col}_glotlid_lang"
    src_prob_col = f"{source_col}_glotlid_prob"
    tgt_lang_col = f"{target_col}_glotlid_lang"
    tgt_prob_col = f"{target_col}_glotlid_prob"

    def _batch_predict(batch):
        src_langs, src_probs = predict(model, batch[source_col])
        tgt_langs, tgt_probs = predict(model, batch[target_col])
        batch[src_lang_col] = src_langs.tolist()
        batch[src_prob_col] = src_probs.tolist()
        batch[tgt_lang_col] = tgt_langs.tolist()
        batch[tgt_prob_col] = tgt_probs.tolist()
        return batch

    return dataset.map(_batch_predict, batched=True, batch_size=batch_size, load_from_cache_file=False)


def annotate_text_units(entries: list[dict], model: fasttext.FastText._FastText | None = None) -> list[dict]:
    """Add text_unit_langs and text_unit_probs to each entry (debug fields).

    Only entries with more than one text unit are run through the model.
    Entries with a single or empty text_units list get None for both fields.
    """
    if model is None:
        model = load_model()

    indexed_texts = [
        (i, text)
        for i, item in enumerate(entries)
        for text in item["text_units"]
        if len(item["text_units"]) > 1
    ]

    if indexed_texts:
        indices, texts = zip(*indexed_texts)
        result = detect_for_texts(list(texts), model)
        result["entry_index"] = list(indices)
        for i, group in result.groupby("entry_index"):
            entries[i]["text_unit_langs"] = group["predicted_language"].reset_index(drop=True).tolist()
            entries[i]["text_unit_probs"] = group["predicted_probability"].reset_index(drop=True).tolist()

    for item in entries:
        item.setdefault("text_unit_langs", None)
        item.setdefault("text_unit_probs", None)

    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate a HuggingFace dataset with GlotLID language predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-repo",
        required=True,
        help="HF Hub dataset repo to load (e.g. madoss/nllb-mos-raw).",
    )
    parser.add_argument(
        "--hub-repo",
        required=True,
        help="HF Hub dataset repo to push annotated results to.",
    )
    parser.add_argument(
        "--source-col",
        default="eng_Latn",
        help="Source language column name (default: %(default)s).",
    )
    parser.add_argument(
        "--target-col",
        default="mos_Latn",
        help="Target language column name (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows per batch for dataset.map (default: %(default)s).",
    )
    parser.add_argument("--private", action="store_true", help="Make the HF dataset private.")
    return parser.parse_args()


def main() -> None:
    from datasets import load_dataset

    args = cli()

    print(f"Loading dataset from '{args.source_repo}'…")
    ds = load_dataset(args.source_repo, split="train")

    print("Loading GlotLID model…")
    model = load_model()

    print(f"Annotating {len(ds):,} rows…")
    ds = annotate_dataset(
        ds, model=model, source_col=args.source_col, target_col=args.target_col, batch_size=args.batch_size
    )

    print(f"Pushing to '{args.hub_repo}'…")
    ds.push_to_hub(args.hub_repo, private=args.private)
    print(f"Done. https://huggingface.co/datasets/{args.hub_repo}")


if __name__ == "__main__":
    main()
