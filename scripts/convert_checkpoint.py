#!/usr/bin/env python3
"""Convert a CaptionStew CLAP *trainer* checkpoint into an inference-ready model.

Trainer checkpoints (``checkpoint-XXXX.pt``, ~3 GB) hold the model, the running
average of the model, optimizer / scheduler / sampler states. This script keeps
only the model weights and writes a self-contained model directory::

    <output_dir>/
      config.json          # ClapConfig (architecture + front-end + text settings)
      model.safetensors    # weights (~775 MB in float32)
      tokenizer files      # RoBERTa-base BPE tokenizer
      README.md            # minimal model card

which can be loaded with ``CaptionStewCLAP.from_pretrained(<output_dir>)`` and
optionally pushed to the Hugging Face Hub (``--push-to-hub``).

Examples
--------
Convert a local checkpoint::

    python scripts/convert_checkpoint.py \
        --checkpoint exp/checkpoint-600000.pt --output-dir models/clap-scratch

Stream a checkpoint directly out of the released tarballs on Hugging Face
(only the bytes up to that checkpoint are downloaded)::

    python scripts/convert_checkpoint.py --from-release scratch --output-dir models/clap-scratch
    python scripts/convert_checkpoint.py --from-release pretrained --output-dir models/clap-pretrained

Average the weights between two trainer checkpoints with their stored running
averages (icefall / Auden ``--iter X --avg N`` style)::

    python scripts/convert_checkpoint.py \
        --checkpoint exp/checkpoint-600000.pt --average-from exp/checkpoint-560000.pt \
        --output-dir models/clap-scratch-avg
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from typing import Dict, Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from captionstew_clap import CaptionStewCLAP, ClapConfig  # noqa: E402

RELEASE_REPO = "RayTseng0406/CaptionStew_checkpoints"
RELEASES = {
    # variant: (tarball in RELEASE_REPO, default checkpoint member pattern)
    "scratch": ("CLAP_from_scratch.tar.gz", "*/checkpoint-600000.pt"),
    "pretrained": ("CLAP_from_pretrained.tar.gz", "*/checkpoint-207000.pt"),
}

MODEL_CARD = """---
library_name: pytorch
tags:
- audio
- clap
- audio-text-retrieval
- zero-shot-audio-classification
datasets:
- RayTseng0406/CaptionStew
---

# {name}

CaptionStew CLAP ({variant}): a contrastive audio-text model trained on
[CaptionStew](https://huggingface.co/datasets/RayTseng0406/CaptionStew) (10M captions), from
[Revisiting Audio-language Pretraining for Learning General-purpose Audio Representation](https://arxiv.org/abs/2511.16757).

* Audio encoder: Zipformer-M on 80-dim log-Mel fbank (16 kHz)
* Text encoder: RoBERTa-base
* Shared embedding size: {dim}
* Source: `{source}` ({weights})

Use it with the [CaptionStew-CLAP](https://github.com/RayTzeng/CaptionStew-CLAP) inference code:

```python
from captionstew_clap import CaptionStewCLAP

model = CaptionStewCLAP.from_pretrained("<this repo or a local dir>")
sim = model.similarity(["dog.wav"], ["a dog barks", "a man is speaking"])
```
"""


def _stream_member_from_release(variant: str, member: Optional[str], workdir: str) -> Dict[str, str]:
    """Stream the release tarball and extract the requested checkpoint (+ config.json).

    The tarball is read sequentially and reading stops right after the checkpoint,
    so only a prefix of the (20-45 GB) archive is downloaded.
    """
    from huggingface_hub import HfFileSystem

    tar_name, default_pattern = RELEASES[variant]
    pattern = member or default_pattern
    fs = HfFileSystem()
    path = f"datasets/{RELEASE_REPO}/{tar_name}"
    logging.info(f"Streaming hf://{path}, looking for '{pattern}'")
    found: Dict[str, str] = {}
    with fs.open(path, "rb", block_size=16 * 1024 * 1024) as f:
        with tarfile.open(fileobj=f, mode="r|gz") as tar:
            for info in tar:
                if not info.isfile():
                    continue
                logging.info(f"  member: {info.name} ({info.size / 1e9:.2f} GB)")
                is_ckpt = fnmatch.fnmatch(info.name, pattern)
                is_cfg = os.path.basename(info.name) == "config.json"
                if not (is_ckpt or is_cfg):
                    continue
                out = os.path.join(workdir, os.path.basename(info.name))
                src = tar.extractfile(info)
                with open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                found["checkpoint" if is_ckpt else "config"] = out
                if is_ckpt:
                    break
    if "checkpoint" not in found:
        raise FileNotFoundError(f"No member matching '{pattern}' in {tar_name}")
    return found


def _load_weights(ckpt_path: str, which: str) -> Dict[str, torch.Tensor]:
    logging.info(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and which in ckpt and isinstance(ckpt[which], dict):
        logging.info(f"Using '{which}' weights (batch_idx_train={ckpt.get('batch_idx_train')})")
        return ckpt[which]
    if isinstance(ckpt, dict) and "model" not in ckpt and "model_avg" not in ckpt:
        logging.info("Checkpoint is a plain state dict")
        return ckpt.get("state_dict", ckpt)
    raise KeyError(f"'{which}' not found in {ckpt_path}; keys: {list(ckpt.keys())}")


def _average(start_path: str, end_path: str) -> Dict[str, torch.Tensor]:
    """Average of the model over the steps between two checkpoints (start excluded).

    Uses the running averages stored as ``model_avg``:
    avg = (model_avg_end * end - model_avg_start * start) / (end - start)
    """
    start = torch.load(start_path, map_location="cpu", weights_only=False)
    end = torch.load(end_path, map_location="cpu", weights_only=False)
    s, e = start["batch_idx_train"], end["batch_idx_train"]
    assert e > s, f"--average-from must be an earlier checkpoint ({s} >= {e})"
    w_end = e / (e - s)
    w_start = 1.0 - w_end
    logging.info(f"Averaging model over steps ({s}, {e}]")
    avg = end["model_avg"]
    for k, v in avg.items():
        if torch.is_floating_point(v):
            avg[k] = (v.double() + start["model_avg"][k].double() * (w_start / w_end)) * w_end
            avg[k] = avg[k].to(v.dtype)
    return avg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Local trainer checkpoint (checkpoint-XXXX.pt)")
    src.add_argument(
        "--from-release",
        choices=sorted(RELEASES),
        help=f"Stream the checkpoint out of the tarball in {RELEASE_REPO}",
    )
    p.add_argument("--member", help="Glob of the tar member to extract with --from-release")
    p.add_argument("--config", help="Training config.json (default: next to the checkpoint, else built-in)")
    p.add_argument(
        "--weights",
        choices=["model", "model_avg"],
        default="model",
        help="Which weights of the trainer checkpoint to export (default: model, the final weights; "
        "model_avg is the running average over all training steps)",
    )
    p.add_argument("--average-from", help="Earlier trainer checkpoint; export the average between the two")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--tokenizer", default="FacebookAI/roberta-base")
    p.add_argument("--keep-download", action="store_true", help="Keep the extracted trainer checkpoint")
    p.add_argument("--push-to-hub", metavar="REPO_ID", help="Upload the converted model to this HF model repo")
    p.add_argument("--private", action="store_true", help="Create the Hub repo as private")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Extract next to the output (the trainer checkpoint is ~3 GB; /tmp may be small)
    parent = os.path.dirname(os.path.abspath(args.output_dir))
    os.makedirs(parent, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="captionstew_ckpt_", dir=parent)
    try:
        if args.from_release:
            found = _stream_member_from_release(args.from_release, args.member, workdir)
            ckpt_path, cfg_path = found["checkpoint"], args.config or found.get("config")
            source = f"{RELEASE_REPO}/{RELEASES[args.from_release][0]}:{os.path.basename(ckpt_path)}"
        else:
            ckpt_path = args.checkpoint
            side = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "config.json")
            cfg_path = args.config or (side if os.path.exists(side) else None)
            source = os.path.basename(ckpt_path)

        config = ClapConfig.from_pretrained(cfg_path) if cfg_path else ClapConfig()
        config.tokenizer_name = args.tokenizer
        if cfg_path is None:
            logging.info("No config.json found; using the default CaptionStew CLAP architecture")

        if args.average_from:
            state = _average(args.average_from, ckpt_path)
            weights = f"average of model_avg between {os.path.basename(args.average_from)} and {source}"
        else:
            state = _load_weights(ckpt_path, args.weights)
            weights = f"'{args.weights}' weights"

        model = CaptionStewCLAP(config)
        model.load_state_dict(state, strict=True)  # validates every key and shape
        model.to(getattr(torch, args.dtype))
        model.load_tokenizer()
        model.save_pretrained(args.output_dir)

        variant = args.from_release or "custom"
        with open(os.path.join(args.output_dir, "README.md"), "w") as f:
            f.write(
                MODEL_CARD.format(
                    name=os.path.basename(os.path.normpath(args.output_dir)),
                    variant={"scratch": "trained from scratch", "pretrained": "AudioSet-initialized audio encoder"}.get(variant, variant),
                    dim=config.shared_emb_dim,
                    source=source,
                    weights=weights,
                )
            )
        logging.info(f"Saved converted model to {args.output_dir}")

        if args.push_to_hub:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(args.push_to_hub, repo_type="model", private=args.private, exist_ok=True)
            api.upload_folder(folder_path=args.output_dir, repo_id=args.push_to_hub, repo_type="model")
            logging.info(f"Uploaded to https://huggingface.co/{args.push_to_hub}")
    finally:
        if args.keep_download and args.from_release:
            logging.info(f"Kept extracted files in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
