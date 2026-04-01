"""Language identification using the GlotLID fasttext model from HuggingFace."""

import fasttext
import pandas as pd
from huggingface_hub import hf_hub_download

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
    scores = pd.Series([p[0] for p in probs], name="predicted_probability")
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

    Adds four new columns:
      source_glotlid_lang, source_glotlid_prob,
      target_glotlid_lang, target_glotlid_prob.
    """
    if model is None:
        model = load_model()

    def _batch_predict(batch):
        src_langs, src_probs = predict(model, batch[source_col])
        tgt_langs, tgt_probs = predict(model, batch[target_col])
        batch["source_glotlid_lang"] = src_langs.tolist()
        batch["source_glotlid_prob"] = src_probs.tolist()
        batch["target_glotlid_lang"] = tgt_langs.tolist()
        batch["target_glotlid_prob"] = tgt_probs.tolist()
        return batch

    return dataset.map(_batch_predict, batched=True, batch_size=batch_size)


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
