# CaptionStew-CLAP: Inference Code for the CaptionStew CLAP Models

This repository provides inference code for the contrastive audio-language (CLAP) models trained on
**CaptionStew**, introduced in our ACL 2026 Long paper
[Revisiting Audio-language Pretraining for Learning General-purpose Audio Representation](https://arxiv.org/abs/2511.16757).
With it you can:

- compute **audio-text similarities** for retrieval and **zero-shot classification**,
- extract **clip embeddings** (1024-d, shared audio-text space) and **frame-level Zipformer features** for downstream
  tasks (linear probing, attention pooling, connecting to an LLM, ...),
- **evaluate audio-text retrieval** on Lhotse manifests or simple CSV/JSONL files,
- **convert** the released trainer checkpoints into small, self-contained model folders (and push them to the Hub).

The training data is described and can be rebuilt with
[CaptionStew-data-prep](https://github.com/RayTzeng/CaptionStew-data-prep). The models were trained with an early
version of [Auden](https://github.com/AudenAI/Auden). This repository re-implements only what inference needs, and
loads the original checkpoints without changing any weights.

## Models

| Model | Audio encoder init | Text encoder init | Steps | Released checkpoint |
|---|---|---|---:|---|
| **Contrastive-scratch** | random | random | 600k | [`CLAP_from_scratch.tar.gz`](https://huggingface.co/datasets/RayTseng0406/CaptionStew_checkpoints/tree/main) → `checkpoint-600000.pt` |
| **Contrastive-init** | Zipformer AudioSet classifier (mAP 0.46) | `roberta-base` | 207k | [`CLAP_from_pretrained.tar.gz`](https://huggingface.co/datasets/RayTseng0406/CaptionStew_checkpoints/tree/main) → `checkpoint-207000.pt` |

Both models share one architecture (193M parameters):

- **Audio encoder**: Zipformer-M (6 stacks, dims 192-256-384-512-384-256, output 512-d at 25 Hz) on 80-dim log-Mel
  filterbanks (16 kHz, 25 ms window, 10 ms hop).
- **Text encoder**: RoBERTa-base (50k BPE vocabulary). The `<s>` token is the sentence representation.
- **Projection heads**: `Linear -> ReLU -> Linear` into a shared 1024-d space. Audio frames are projected first and then
  mean-pooled over time. Both embeddings are L2-normalized; the training temperature was 0.07.

Sanity check of the converted models: retrieval on a 761-clip / 1,467-caption subset of the AudioCaps test set (the first
shard of [`AudioLLMs/audiocaps_test`](https://huggingface.co/datasets/AudioLLMs/audiocaps_test)), computed with
`scripts/evaluate_retrieval.py`. This is not the official AudioCaps protocol and is not comparable to the numbers in the
paper.

| Model | T→A R@1 | T→A R@10 | A→T R@1 | A→T R@10 |
|---|---:|---:|---:|---:|
| Contrastive-scratch | 39.5 | 85.3 | 43.4 | 86.6 |
| Contrastive-init | 41.4 | 88.2 | 46.6 | 87.4 |

## Installation

    git clone https://github.com/RayTzeng/CaptionStew-CLAP
    cd CaptionStew-CLAP
    pip install -e .            # core: torch, transformers, safetensors, soundfile, ...
    pip install -e ".[audio]"   # optional: torchaudio (faster resampling, mp3/m4a fallback)
    pip install -e ".[lhotse]"  # optional: evaluate on Lhotse manifests

[k2](https://github.com/k2-fsa/k2) is **not** required. The vendored Zipformer falls back to pure-PyTorch activations
when k2 is missing.

## Getting a model

The released files on Hugging Face are full *trainer* checkpoints (~3 GB each, holding the model, a running average of
the model, and the optimizer, scheduler and sampler states) packed in large tarballs. `scripts/convert_checkpoint.py`
turns one into a self-contained model folder:

    models/clap-scratch/
    ├── config.json         # architecture, front-end and text settings
    ├── model.safetensors   # weights (~775 MB in float32)
    ├── tokenizer/          # RoBERTa-base tokenizer
    └── README.md           # small model card

**Straight from the release.** The tarball is read as a stream and reading stops right after the checkpoint, so only
about 3 GB of the archive is downloaded:

    python scripts/convert_checkpoint.py --from-release scratch    --output-dir models/clap-scratch
    python scripts/convert_checkpoint.py --from-release pretrained --output-dir models/clap-pretrained

**From a local trainer checkpoint** (e.g. after extracting a tarball). A `config.json` next to the checkpoint is
picked up automatically:

    python scripts/convert_checkpoint.py --checkpoint path/to/checkpoint-600000.pt --output-dir models/clap-scratch

Useful options:

- `--weights {model,model_avg}`: export the final weights (`model`, default) or the running average over all training
  steps (`model_avg`). On the AudioCaps subset above, `model` scored 1–1.5 R@1 points higher for the scratch model.
- `--average-from path/to/checkpoint-560000.pt`: export the model averaged over the steps between two trainer checkpoints
  (icefall/Auden `--iter X --avg N` style averaging).
- `--dtype float16`: halve the size of `model.safetensors`.
- `--push-to-hub <user>/<repo> [--private]`: upload the converted folder as a Hugging Face model repo (run
  `hf auth login` first). After that, `CaptionStewCLAP.from_pretrained("<user>/<repo>")` downloads it directly.

## Usage

### Python API

```python
import torch
from captionstew_clap import CaptionStewCLAP

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CaptionStewCLAP.from_pretrained("models/clap-pretrained", device=device)  # or a HF model repo id

with torch.inference_mode():
    # Audio: file paths, 1-D waveforms (16 kHz numpy/torch), or precomputed (fbank, fbank_lens)
    audio_emb = model.encode_audio(["dog.wav", "street.flac"])              # (2, 1024), L2-normalized
    text_emb = model.encode_text(["a dog barks", "cars pass by on a busy street"])  # (2, 1024)
    sim = audio_emb @ text_emb.T                                           # cosine similarities

    # Shortcut
    sim = model.similarity(["dog.wav"], ["a dog barks", "a woman is singing"])
```

**Zero-shot classification.** Embed one text prompt per class and pick the most similar:

```python
labels = ["dog barking", "speech", "music", "rain", "siren"]
with torch.inference_mode():
    probs = (model.similarity(["clip.wav"], labels) / model.config.temp).softmax(dim=-1)
```

**Audio representations.** The paper evaluates the frozen audio encoder's final layer:

```python
with torch.inference_mode():
    out = model.audio_frame_features(["clip.wav"], all_layers=True)
frames = out["encoder_out"]          # (B, T', 512) at 25 Hz, valid lengths in out["encoder_out_lens"]
stacks = out["layer_outputs"]        # outputs of the 6 Zipformer stacks, (B, T', C_i) each
clip = model.encode_audio(["clip.wav"], normalize=False)  # pooled + projected, before L2 norm
```

Things to keep in mind:

- Captions are **lower-cased and stripped of punctuation** before tokenization, exactly as during training
  (`captionstew_clap.preprocess_caption`). `encode_text` does this for you; pass `preprocess=False` to skip it.
- Training clips were **1–32 s** long (longer clips were truncated to 32 s). For longer recordings, embed 10–30 s windows and average the embeddings.
- Batching pads the fbank features, which changes embeddings very slightly (cosine difference < 1e-3) compared with
  encoding each clip alone. Training used padded batches too.
- Audio files are down-mixed to mono and resampled to 16 kHz automatically. Waveform arrays must already be 16 kHz
  mono.

### Command line

Rank captions or class labels for audio files and save embeddings:

    python scripts/inference.py --model models/clap-pretrained \
        --audio dog.wav street.flac \
        --text "a dog barks" "cars pass by on a busy street" "a woman is singing"

    python scripts/inference.py --model models/clap-pretrained --audio clips/*.wav \
        --labels labels.txt --template "{}" --top-k 3 --save-embeddings embeddings.pt

### Retrieval evaluation

`scripts/evaluate_retrieval.py` reports text→audio and audio→text R@1/5/10 and mAP@10, with any number of captions per
clip. Every caption is a query against all clips, and every clip is a query against all captions (any of its own
captions counts as a hit).

    # Lhotse CutSet manifests in the CaptionStew-data-prep format (captions in supervisions[0].custom.caption)
    python scripts/evaluate_retrieval.py --model models/clap-pretrained \
        --manifest my_test_cuts.jsonl.gz --output metrics.json

    # CSV / TSV / JSONL with "audio" and "caption" columns (rows of the same audio are grouped)
    python scripts/evaluate_retrieval.py --model models/clap-pretrained --manifest audiocaps_test.csv

## Implementation notes

- `captionstew_clap/features.py` re-implements, in plain PyTorch, the Lhotse `Fbank(FbankConfig(num_mel_bins=80))`
  front-end used in training (povey window, pre-emphasis 0.97, DC removal, `snip_edges=False`, no dither). It matches
  Lhotse to within 1e-4 (see `tests/`).
- The parameter names of `CaptionStewCLAP` (`encoder_embed`, `encoder`, `text_encoder`, `audio_proj`, `text_proj`) match
  the training checkpoints, so they load with `strict=True`. The public Auden `ClapModel` names its audio tower
  `audio_encoder.*` and builds BERT instead of RoBERTa, so the released checkpoints do not load into it directly.
- We confirmed the pooling and text conventions empirically on AudioCaps. Projecting frames before mean pooling, the
  `<s>` token, and caption preprocessing each gave clearly higher retrieval scores than the alternatives.

## Repository layout

    captionstew_clap/
      model.py            CaptionStewCLAP: loading, encode_audio / encode_text / audio_frame_features
      config.py           ClapConfig (also reads the flat config.json written by the training code)
      features.py         audio loading, resampling, Kaldi-compatible fbank
      text.py             caption preprocessing
      zipformer/          Zipformer encoder (vendored from Auden / icefall)
    scripts/
      convert_checkpoint.py   trainer checkpoint / release tarball -> model folder (+ Hub upload)
      inference.py            similarity, zero-shot labels, embedding export
      evaluate_retrieval.py   retrieval metrics on Lhotse or CSV/JSONL manifests
    tests/                    unit tests (no download needed): pytest tests

## Citation

If you find this repository useful, please consider citing the following work:
```bibtex
@article{tseng2025revisiting,
  title={Revisiting Audio-language Pretraining for Learning General-purpose Audio Representation},
  author={Tseng, Wei-Cheng and Zhou, Xuanru and Huo, Mingyue and Shao, Yiwen and Zhang, Hao and Yu, Dong},
  journal={arXiv preprint arXiv:2511.16757},
  year={2025}
}
```

## Acknowledgements

The Zipformer encoder and the CLAP recipe come from [Auden](https://github.com/AudenAI/Auden) and
[icefall](https://github.com/k2-fsa/icefall), and the feature front-end follows [Lhotse](https://github.com/lhotse-speech/lhotse),
all under the Apache License 2.0. See [NOTICE](NOTICE).

## Disclaimer

This repository has not yet been extensively tested. If you run into a problem or have a suggestion, please
[open an issue](https://github.com/RayTzeng/CaptionStew-CLAP/issues) or submit a pull request.
