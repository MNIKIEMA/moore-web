"""Language identification using the MEAG LID model from HuggingFace."""

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "JessicaOjo/meag_lid"
FILENAME = "model/model.joblib"


def load_model(repo_id: str = REPO_ID) -> dict:
    """Download and load the NB model bundle from HuggingFace Hub."""
    model_path = hf_hub_download(repo_id=repo_id, filename=FILENAME, repo_type="model")
    return joblib.load(model_path)


def predict(nb_bundle: dict, texts: list[str]) -> tuple[pd.Series, pd.Series]:
    """Return (predicted_language, predicted_probability) series for each text."""
    clf = nb_bundle["model"]
    vec = nb_bundle["vectorizer"]
    X_vec = vec.transform(texts)
    langs = pd.Series(clf.predict(X_vec), name="predicted_language")
    probs = pd.Series(clf.predict_proba(X_vec).max(axis=1), name="predicted_probability")
    return langs, probs


def detect_for_texts(texts: list[str], nb_bundle: dict | None = None) -> pd.DataFrame:
    """Run lang ID on a flat list of strings. Returns a DataFrame with columns:
    text, predicted_language, predicted_probability.
    """
    if nb_bundle is None:
        nb_bundle = load_model()
    langs, probs = predict(nb_bundle, texts)
    return pd.DataFrame({"text": texts, "predicted_language": langs, "predicted_probability": probs})


def annotate_text_units(entries: list[dict], nb_bundle: dict | None = None) -> list[dict]:
    """Add text_unit_langs and text_unit_probs to each entry (debug fields).

    Only entries with more than one text unit are run through the model.
    Entries with a single or empty text_units list get None for both fields.
    """
    if nb_bundle is None:
        nb_bundle = load_model()

    indexed_texts = [
        (i, text)
        for i, item in enumerate(entries)
        for text in item["text_units"]
        if len(item["text_units"]) > 1
    ]

    if indexed_texts:
        indices, texts = zip(*indexed_texts)
        result = detect_for_texts(list(texts), nb_bundle)
        result["entry_index"] = list(indices)
        for i, group in result.groupby("entry_index"):
            entries[i]["text_unit_langs"] = group["predicted_language"].reset_index(drop=True).tolist()
            entries[i]["text_unit_probs"] = group["predicted_probability"].reset_index(drop=True).tolist()

    for item in entries:
        item.setdefault("text_unit_langs", None)
        item.setdefault("text_unit_probs", None)

    return entries
