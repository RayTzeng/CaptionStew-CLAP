"""Configuration for CaptionStew CLAP models."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List

# Architecture of RoBERTa-base, embedded so that the model can be built offline.
ROBERTA_BASE_CONFIG: Dict[str, Any] = {
    "model_type": "roberta",
    "vocab_size": 50265,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 514,
    "type_vocab_size": 1,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-05,
    "pad_token_id": 1,
    "bos_token_id": 0,
    "eos_token_id": 2,
}


@dataclass
class ZipformerConfig:
    """Zipformer-M audio encoder (the defaults are the CaptionStew setup)."""

    feature_dim: int = 80
    output_downsampling_factor: int = 2
    num_encoder_layers: List[int] = field(default_factory=lambda: [2, 2, 3, 4, 3, 2])
    downsampling_factor: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 4, 2])
    encoder_dim: List[int] = field(default_factory=lambda: [192, 256, 384, 512, 384, 256])
    feedforward_dim: List[int] = field(
        default_factory=lambda: [576, 768, 1152, 1536, 1152, 768]
    )
    num_heads: List[int] = field(default_factory=lambda: [4, 4, 4, 8, 4, 4])
    query_head_dim: List[int] = field(default_factory=lambda: [32])
    value_head_dim: List[int] = field(default_factory=lambda: [12])
    pos_head_dim: List[int] = field(default_factory=lambda: [4])
    pos_dim: int = 48
    encoder_unmasked_dim: List[int] = field(
        default_factory=lambda: [192, 192, 256, 256, 256, 192]
    )
    cnn_module_kernel: List[int] = field(default_factory=lambda: [31, 31, 15, 15, 15, 31])
    causal: bool = False
    chunk_size: List[int] = field(default_factory=lambda: [16, 32, 64, -1])
    left_context_frames: List[int] = field(default_factory=lambda: [64, 128, 256, -1])
    warmup_batches: float = 4000.0
    dropout: float | None = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ZipformerConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class ClapConfig:
    """Full CLAP model: Zipformer audio tower + RoBERTa text tower + projections."""

    model_type: str = "captionstew-clap"
    audio_encoder: ZipformerConfig = field(default_factory=ZipformerConfig)
    text_encoder: Dict[str, Any] = field(default_factory=lambda: dict(ROBERTA_BASE_CONFIG))
    tokenizer_name: str = "FacebookAI/roberta-base"
    shared_emb_dim: int = 1024
    temp: float = 0.07
    # Front-end (must match training: 16 kHz, 80-dim Kaldi-style log-Mel fbank)
    sampling_rate: int = 16000
    num_mel_bins: int = 80
    # Text side
    max_text_length: int = 512
    lowercase_and_strip_punctuation: bool = True

    # ------------------------------------------------------------------ io
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClapConfig":
        d = copy.deepcopy(d)
        if "audio_encoder" not in d and "encoder_dim" in d:
            # Flat config.json written by the original training code
            # (model name "zipformer-clap"). Convert it on the fly.
            return cls.from_training_config(d)
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}
        if isinstance(kwargs.get("audio_encoder"), dict):
            kwargs["audio_encoder"] = ZipformerConfig.from_dict(kwargs["audio_encoder"])
        return cls(**kwargs)

    @classmethod
    def from_training_config(cls, d: Dict[str, Any]) -> "ClapConfig":
        """Build from the flat ``config.json`` saved next to trainer checkpoints."""
        text_type = d.get("text_encoder_type", "roberta-base")
        if text_type not in ("roberta-base", "FacebookAI/roberta-base"):
            raise ValueError(
                f"Unsupported text encoder '{text_type}'; CaptionStew CLAP uses roberta-base."
            )
        if d.get("pooling", "mean") != "mean":
            raise ValueError(f"Unsupported audio pooling '{d['pooling']}' (only 'mean').")
        return cls(
            audio_encoder=ZipformerConfig.from_dict(d),
            shared_emb_dim=d.get("shared_emb_dim", 1024),
            temp=d.get("temp", 0.07),
        )

    def save_pretrained(self, save_directory: str) -> str:
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def from_pretrained(cls, path: str) -> "ClapConfig":
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
