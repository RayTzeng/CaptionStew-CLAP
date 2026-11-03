#!/usr/bin/env python3
"""Audio-text similarity, zero-shot classification and embedding extraction.

Examples
--------
Rank captions for each audio file::

    python scripts/inference.py --model models/clap-scratch \
        --audio dog.wav street.flac \
        --text "a dog barks" "cars passing by on a busy street" "a woman is singing"

Zero-shot classification with a label list (one label per line)::

    python scripts/inference.py --model models/clap-scratch --audio clips/*.wav \
        --labels labels.txt --template "the sound of {}"

Save embeddings (audio, text, or both) for later use::

    python scripts/inference.py --model models/clap-scratch --audio clips/*.wav \
        --save-embeddings embeddings.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from captionstew_clap import CaptionStewCLAP  # noqa: E402


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Converted model directory or HF model repo id")
    p.add_argument("--audio", nargs="*", default=[], help="Audio files")
    p.add_argument("--text", nargs="*", default=[], help="Text queries / captions")
    p.add_argument("--labels", help="File with one class label per line (zero-shot classification)")
    p.add_argument("--template", default="{}", help="Prompt template for --labels, e.g. 'the sound of {}'")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-embeddings", help="Write {'audio': ..., 'text': ...} embeddings to this .pt file")
    args = p.parse_args()

    texts = list(args.text)
    if args.labels:
        with open(args.labels) as f:
            labels = [line.strip() for line in f if line.strip()]
        texts += [args.template.format(lab) for lab in labels]
    if not args.audio and not texts:
        p.error("give --audio and/or --text/--labels")

    model = CaptionStewCLAP.from_pretrained(args.model, device=args.device)

    with torch.inference_mode():
        a_emb = (
            torch.cat([model.encode_audio(b).cpu() for b in batched(args.audio, args.batch_size)])
            if args.audio
            else None
        )
        t_emb = torch.cat([model.encode_text(b).cpu() for b in batched(texts, 64)]) if texts else None

    if args.save_embeddings:
        torch.save(
            {"audio": a_emb, "audio_files": args.audio, "text": t_emb, "texts": texts},
            args.save_embeddings,
        )
        print(f"Saved embeddings to {args.save_embeddings}")

    if a_emb is not None and t_emb is not None:
        sim = a_emb @ t_emb.T
        k = min(args.top_k, len(texts))
        for path, row in zip(args.audio, sim):
            print(f"\n{path}")
            probs = (row / model.config.temp).softmax(dim=-1)
            scores, idx = row.topk(k)
            for s, i in zip(scores.tolist(), idx.tolist()):
                print(f"  {s:+.3f}  p={probs[i]:.3f}  {texts[i]}")


if __name__ == "__main__":
    main()
