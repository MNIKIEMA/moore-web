"""Translation utilities using a local vLLM OpenAI-compatible server.

Usage
-----
    # Start vLLM server first:
    vllm serve tencent/HY-MT1.5-7B --served-model-name hy-mt --port 8021 --max-model-len 32768

    # Translate English segments to French and push to HF Hub:
    python -m moore_web.translation --source-repo madoss/nllb-mos-filtered --hub-repo madoss/nllb-mos-fr
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv
from moore_web.upload_nllb_raw import _COLS as _TSV_COLS, _load_nllb_tsv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

load_dotenv()

_SOURCE_COL = "eng_Latn"
_TARGET_COL = "eng_Latn_to_fra_Latn"
_INSTRUCTION = "Translate the following segment into French, without additional explanation."


async def translate(
    sentences: list[str],
    model: str = "hy-mt",
    base_url: str = "http://localhost:8021/v1",
    concurrency: int = 128,
) -> list[str]:
    """Translate a list of sentences to French via a local vLLM server.

    Args:
        sentences:   Texts to translate.
        model:       Model name as served by the vLLM server.
        base_url:    Base URL of the OpenAI-compatible vLLM server.
        concurrency: Maximum number of in-flight requests.

    Returns:
        List of translated strings in the same order as ``sentences``.

    Example::

        import asyncio
        from moore_web.translation import translate

        fra = asyncio.run(translate(["Hello world"]))
    """
    client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)
    semaphore = asyncio.Semaphore(concurrency)

    async def _translate_one(text: str) -> str:
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _INSTRUCTION},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.0,
                    top_p=0.9,
                    extra_body={
                        "top_k": 20,
                        "repetition_penalty": 1.05,
                    },
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"\nError translating example: {e}")
                return ""

    tasks = [_translate_one(text) for text in sentences]
    return await tqdm.gather(*tasks, desc="Translating")


async def translate_and_upload(
    hub_repo: str = "madoss/nllb-mos-fr",
    source_repo: str | None = None,
    model: str = "hy-mt",
    base_url: str = "http://localhost:8021/v1",
    concurrency: int = 32,
    source_col: str = _SOURCE_COL,
    target_col: str = _TARGET_COL,
    private: bool = False,
) -> None:
    """Translate English segments to French and push the enriched dataset to HF Hub.

    Loads from ``source_repo`` if provided, otherwise re-downloads the raw TSV.
    Adds a ``target_col`` column with the translations.

    Args:
        hub_repo:     HuggingFace repo to push the translated dataset to.
        source_repo:  HF Hub repo to load the source dataset from.
                      If None, the raw TSV is downloaded directly.
        model:        Model name as served by the vLLM server.
        base_url:     Base URL of the OpenAI-compatible vLLM server.
        concurrency:  Maximum number of in-flight requests.
        source_col:   Column name containing source sentences.
        target_col:   Column name to write translations into.
        private:      Whether to make the HF Hub dataset private.
    """
    from datasets import Dataset, DatasetDict, load_dataset

    if source_repo:
        print(f"Loading dataset from HuggingFace Hub: '{source_repo}'…")
        ds = load_dataset(source_repo, split="train")
        rows = [dict(zip(ds.column_names, vals)) for vals in zip(*[ds[col] for col in ds.column_names])]
    else:
        rows = _load_nllb_tsv()

    eng_sentences = [r[source_col] for r in rows]

    fra_sentences = await translate(eng_sentences, model=model, base_url=base_url, concurrency=concurrency)

    cols = list(rows[0].keys()) if rows else _TSV_COLS
    dataset = Dataset.from_dict(
        {
            **{col: [r[col] for r in rows] for col in cols},
            target_col: fra_sentences,
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
        description="Translate English segments to French via a local vLLM server and push to HF Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hub-repo",
        default="madoss/nllb-mos-fr",
        help="HuggingFace Hub repo to push the translated dataset to (default: %(default)s).",
    )
    parser.add_argument(
        "--source-repo",
        default="madoss/nllb-mos-filtered",
        help="HF Hub repo to load the source dataset from. If omitted, re-downloads the TSV.",
    )
    parser.add_argument(
        "--model",
        default="hy-mt",
        help="Model name as served by the vLLM server (default: %(default)s).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8021/v1",
        help="Base URL of the OpenAI-compatible vLLM server (default: %(default)s).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=128,
        help="Maximum number of in-flight requests (default: %(default)s).",
    )
    parser.add_argument(
        "--source-col",
        default=_SOURCE_COL,
        help="Dataset column containing source sentences (default: %(default)s).",
    )
    parser.add_argument(
        "--target-col",
        default=_TARGET_COL,
        help="Dataset column to write translations into (default: %(default)s).",
    )
    parser.add_argument("--private", action="store_true", help="Make the dataset private.")

    args = parser.parse_args()
    asyncio.run(translate_and_upload(
        hub_repo=args.hub_repo,
        source_repo=args.source_repo,
        model=args.model,
        base_url=args.base_url,
        concurrency=args.concurrency,
        source_col=args.source_col,
        target_col=args.target_col,
        private=args.private,
    ))


if __name__ == "__main__":
    main()
