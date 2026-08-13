#!/usr/bin/env python3
# flores_qe.py

import argparse
import numpy as np
from datasets import load_dataset
from comet import download_model, load_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="COMET-QE scoring on FLORES+ language pairs")
    parser.add_argument("--src", default="eng_Latn", help="Source language code (e.g. eng_Latn)")
    parser.add_argument("--tgt", default="mos_Latn", help="Target language code (e.g. mos_Latn)")
    parser.add_argument("--split", default="dev", choices=["dev", "devtest"], help="FLORES+ split to use")
    parser.add_argument("--model", default="McGill-NLP/ssa-comet-qe", help="COMET model to use")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--output", help="Output CSV path (default: comet_qe_{src}_{tgt}.csv)")
    args = parser.parse_args()

    src, tgt = args.src, args.tgt
    output_path = args.output or f"comet_qe_{src}_{tgt}.csv"
    col_name = f"comet_qe_{src}_{tgt}"

    ds_src = load_dataset("openlanguagedata/flores_plus", src, split=args.split)
    ds_tgt = load_dataset("openlanguagedata/flores_plus", tgt, split=args.split)

    ds_src = ds_src.select_columns(["id", "topic", "text"]).rename_column("text", src)
    ds_tgt = ds_tgt.select_columns(["id", "topic", "text"]).rename_column("text", tgt)

    merged = ds_src.add_column(tgt, ds_tgt[tgt])

    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)

    data = [{"src": row[src], "mt": row[tgt]} for row in merged]
    output = model.predict(data, batch_size=args.batch_size, gpus=args.gpus)

    merged = merged.add_column(col_name, output.scores)

    scores = np.array(output.scores)
    print(f"\n=== QE Score Stats ({src} → {tgt}, {len(scores)} sentences) ===")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    print(f"  Std:    {scores.std():.4f}")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Max:    {scores.max():.4f}")

    print("\n=== Per-topic Mean QE Score ===")
    topics = merged["topic"]
    for topic in sorted(set(topics)):
        topic_scores = scores[[i for i, t in enumerate(topics) if t == topic]]
        print(f"  {topic:<30} {topic_scores.mean():.4f}  (n={len(topic_scores)})")

    print("\n=== Score Distribution ===")
    for low, high in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        count = ((scores >= low) & (scores < high)).sum()
        print(f"  [{low:.1f}, {high:.1f}): {count:>4} ({100 * count / len(scores):.1f}%)")

    merged.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
