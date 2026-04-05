"""Translation utilities using TranslateGemma (google/translategemma-*) via vLLM.

Usage
-----
    # Translate English segments to French and push to HF Hub
    uv run python -m moore_web.translation --source-repo madoss/nllb-mos-raw
    uv run python -m moore_web.translation --hub-repo madoss/nllb-mos-fr --model google/translategemma-4b-it
"""

from __future__ import annotations

import argparse

from moore_web.upload_nllb_raw import _COLS as _TSV_COLS, _load_nllb_tsv


def translate(
    sentences: list[str],
    source_lang: str,
    target_lang: str,
    model: str = "google/translategemma-4b-it",
    batch_size: int = 32,
    max_new_tokens: int = 512,
    tensor_parallel_size: int = 1,
) -> list[str]:
    """Translate a list of sentences using TranslateGemma via vLLM.

    Args:
        sentences:            Texts to translate.
        source_lang:          BCP-47 source language code (e.g. ``"en"``).
        target_lang:          BCP-47 target language code (e.g. ``"fr-FR"``).
        model:                HuggingFace model ID for TranslateGemma.
        batch_size:           Sentences per vLLM generation call.
        max_new_tokens:       Maximum tokens to generate per sentence.
        tensor_parallel_size: Number of GPUs for tensor parallelism.

    Returns:
        List of translated strings in the same order as ``sentences``.

    Example::

        from moore_web.translation import translate

        fra = translate(["Hello world"], source_lang="en", target_lang="fr-FR")
    """
    from vllm import LLM, SamplingParams

    print(f"Loading translation model ({model})…")
    llm = LLM(model=model, dtype="bfloat16", tensor_parallel_size=tensor_parallel_size)
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)

    translations: list[str] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_lang,
                            "target_lang_code": target_lang,
                            "text": text,
                        }
                    ],
                }
            ]
            for text in batch
        ]
        outputs = llm.chat(conversations, sampling_params=sampling_params)
        translations.extend(out.outputs[0].text.strip() for out in outputs)
        print(f"Translated {min(i + batch_size, len(sentences))}/{len(sentences)} sentences…")

    return translations


def translate_and_upload(
    hub_repo: str = "madoss/nllb-mos-fr",
    source_repo: str | None = None,
    model: str = "google/translategemma-12b-it",
    batch_size: int = 32,
    max_new_tokens: int = 512,
    tensor_parallel_size: int = 1,
    private: bool = False,
) -> None:
    """Translate English segments to French and push the enriched dataset to HF Hub.

    Loads from ``source_repo`` if provided, otherwise re-downloads the raw TSV.
    Adds a ``fra_Latn`` column with TranslateGemma translations.

    Args:
        hub_repo:             HuggingFace repo to push the translated dataset to.
        source_repo:          HF Hub repo to load the source dataset from.
                              If None, the raw TSV is downloaded directly.
        model:                TranslateGemma model ID.
        batch_size:           Sentences per vLLM generation call.
        max_new_tokens:       Maximum tokens to generate per sentence.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        private:              Whether to make the HF Hub dataset private.
    """
    from datasets import Dataset, DatasetDict, load_dataset

    if source_repo:
        print(f"Loading dataset from HuggingFace Hub: '{source_repo}'…")
        ds = load_dataset(source_repo, split="train")
        rows = [dict(zip(ds.column_names, vals)) for vals in zip(*[ds[col] for col in ds.column_names])]
    else:
        rows = _load_nllb_tsv()

    eng_sentences = [r["eng_Latn"] for r in rows]

    fra_sentences = translate(
        eng_sentences,
        source_lang="en",
        target_lang="fr-FR",
        model=model,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        tensor_parallel_size=tensor_parallel_size,
    )

    cols = list(rows[0].keys()) if rows else _TSV_COLS
    dataset = Dataset.from_dict(
        {
            **{col: [r[col] for r in rows] for col in cols},
            "eng_Latn_to_fra_Latn": fra_sentences,
        }
    )

    dataset_dict = DatasetDict({"train": dataset})

    print(f"\nPushing translated dataset to HuggingFace Hub as '{hub_repo}'…")
    dataset_dict.push_to_hub(hub_repo, private=private)
    print(f"Done. Dataset available at https://huggingface.co/datasets/{hub_repo}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Translate English segments to French and push to HF Hub."""
    parser = argparse.ArgumentParser(
        description="Translate English segments to French with TranslateGemma and push to HF Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hub-repo",
        default="madoss/nllb-mos-filtered",
        help="HuggingFace Hub repo to push the translated dataset to (default: %(default)s).",
    )
    parser.add_argument(
        "--source-repo",
        default="madoss/nllb-mos-filtered",
        help="HF Hub repo to load the source dataset from. If omitted, re-downloads the TSV.",
    )
    parser.add_argument(
        "--model",
        default="google/translategemma-4b-it",
        help="TranslateGemma model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Sentences per vLLM generation call (default: %(default)s).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per sentence (default: %(default)s).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: %(default)s).",
    )
    parser.add_argument("--private", action="store_true", help="Make the dataset private.")

    args = parser.parse_args()
    translate_and_upload(
        hub_repo=args.hub_repo,
        source_repo=args.source_repo,
        model=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        private=args.private,
    )


if __name__ == "__main__":
    main()
