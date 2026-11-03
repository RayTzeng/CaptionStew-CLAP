#!/usr/bin/env python3
"""Audio-text retrieval evaluation (R@1/5/10, mAP@10) with multiple captions per clip.

Input manifests:

* Lhotse CutSet manifests (``*.jsonl`` / ``*.jsonl.gz``) such as those produced by
  `CaptionStew-data-prep <https://github.com/RayTzeng/CaptionStew-data-prep>`_;
  captions are read from ``supervisions[0].custom[<caption-field>]`` (string or list).
  Requires ``pip install lhotse``.
* Simple ``.csv`` / ``.tsv`` / ``.jsonl`` files with an ``audio`` column (path) and a
  ``caption`` column. Rows sharing the same audio path are grouped; in JSONL the
  caption may also be a list.

Every caption is a text->audio query against all clips; every clip is an
audio->text query against all captions (a hit is any of its own captions).

Example::

    python scripts/evaluate_retrieval.py --model models/clap-scratch \
        --manifest manifests/audiocaps_test.jsonl.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import sys
from collections import OrderedDict
from typing import List, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from captionstew_clap import CaptionStewCLAP  # noqa: E402


# ---------------------------------------------------------------------- data
def _is_lhotse_manifest(path: str) -> bool:
    if not (path.endswith(".jsonl") or path.endswith(".jsonl.gz")):
        return False
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        first = json.loads(f.readline())
    return "supervisions" in first or "recording" in first


def _as_list(c) -> List[str]:
    if c is None:
        return []
    return [str(x) for x in c] if isinstance(c, (list, tuple)) else [str(c)]


def load_lhotse(path: str, field: str, sr: int, min_duration: float):
    from lhotse import CutSet

    cuts = CutSet.from_file(path)
    items = []
    for cut in cuts:
        if cut.duration < min_duration or not cut.supervisions:
            continue
        sup = cut.supervisions[0]
        caps = _as_list((sup.custom or {}).get(field, getattr(sup, field, None)))
        if caps:
            items.append((cut, caps))

    def loader(cut):
        if cut.sampling_rate != sr:
            cut = cut.resample(sr)
        audio = torch.from_numpy(cut.load_audio())
        return audio.mean(dim=0) if audio.dim() == 2 else audio

    return [(c.id, c, caps) for c, caps in items], loader


def load_table(path: str, audio_col: str, caption_col: str):
    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    if path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t" if path.endswith(".tsv") else ","))
    base = os.path.dirname(os.path.abspath(path))
    for r in rows:
        a = r[audio_col]
        a = a if os.path.isabs(a) else os.path.join(base, a)
        groups.setdefault(a, []).extend(_as_list(r[caption_col]))
    return [(a, a, caps) for a, caps in groups.items()], (lambda a: a)


# ---------------------------------------------------------------------- metrics
@torch.no_grad()
def retrieval_metrics(audio_emb: torch.Tensor, text_emb: torch.Tensor, cap2clip: torch.Tensor, k: int = 10):
    """Return dict of t2a / a2t R@1,5,10 and mAP@k (percent)."""
    num_clips = audio_emb.size(0)
    # text -> audio: one positive clip per caption
    sim_t2a = text_emb @ audio_emb.T  # (C, A)
    pos = sim_t2a.gather(1, cap2clip[:, None])
    rank_t2a = (sim_t2a > pos).sum(1) + 1  # 1-based
    # audio -> text: several positive captions per clip
    sim_a2t = sim_t2a.T  # (A, C)
    is_pos = cap2clip[None, :] == torch.arange(num_clips)[:, None]  # (A, C)
    order = sim_a2t.argsort(dim=1, descending=True)
    hits = is_pos.gather(1, order)  # (A, C) bool, sorted by score
    first_hit = hits.float().argmax(dim=1) + 1
    topk = hits[:, :k].float()
    precision_at = topk.cumsum(1) / torch.arange(1, topk.size(1) + 1)
    n_pos = is_pos.sum(1).clamp(max=k).float()
    ap_a2t = (precision_at * topk).sum(1) / n_pos

    def recalls(r):
        return {f"R@{n}": 100.0 * (r <= n).float().mean().item() for n in (1, 5, 10)}

    t2a = recalls(rank_t2a)
    t2a[f"mAP@{k}"] = 100.0 * torch.where(rank_t2a <= k, 1.0 / rank_t2a.float(), 0.0).mean().item()
    a2t = recalls(first_hit)
    a2t[f"mAP@{k}"] = 100.0 * ap_a2t.mean().item()
    return {"text_to_audio": t2a, "audio_to_text": a2t}


# ---------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Converted model directory or HF model repo id")
    p.add_argument("--manifest", required=True, nargs="+", help="Lhotse cuts or csv/tsv/jsonl table(s)")
    p.add_argument("--caption-field", default="caption", help="Caption field name")
    p.add_argument("--audio-column", default="audio", help="Audio column for csv/tsv/jsonl tables")
    p.add_argument("--min-duration", type=float, default=1.0, help="Skip shorter Lhotse cuts (as in training)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", help="Write metrics as JSON here")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model = CaptionStewCLAP.from_pretrained(args.model, device=args.device)
    sr = model.config.sampling_rate
    results = {}
    for manifest in args.manifest:
        if _is_lhotse_manifest(manifest):
            items, loader = load_lhotse(manifest, args.caption_field, sr, args.min_duration)
        else:
            items, loader = load_table(manifest, args.audio_column, args.caption_field)
        captions = [c for _, _, caps in items for c in caps]
        cap2clip = torch.tensor([i for i, (_, _, caps) in enumerate(items) for _ in caps])
        logging.info(f"{manifest}: {len(items)} clips, {len(captions)} captions")

        audio_embs: List[torch.Tensor] = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(items), args.batch_size), desc="audio"):
                batch = [loader(src) for _, src, _ in items[i : i + args.batch_size]]
                audio_embs.append(model.encode_audio(batch).float().cpu())
            text_embs = [
                model.encode_text(captions[i : i + 128]).float().cpu()
                for i in tqdm(range(0, len(captions), 128), desc="text")
            ]
        m = retrieval_metrics(torch.cat(audio_embs), torch.cat(text_embs), cap2clip)
        m["num_clips"], m["num_captions"] = len(items), len(captions)
        results[manifest] = m
        for direction in ("text_to_audio", "audio_to_text"):
            line = "  ".join(f"{k} {v:5.1f}" for k, v in m[direction].items())
            print(f"[{os.path.basename(manifest)}] {direction:13s} {line}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
