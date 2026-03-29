"""Translation utilities using TranslateGemma (google/translategemma-*) via vLLM."""

from __future__ import annotations


def translate(
    sentences: list[str],
    source_lang: str,
    target_lang: str,
    model: str = "google/translategemma-12b-it",
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
