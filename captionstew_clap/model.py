"""CaptionStew CLAP: Zipformer audio encoder + RoBERTa text encoder."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ClapConfig
from .features import AudioInput, Fbank, compute_fbank
from .text import preprocess_caption
from .zipformer.padding import make_pad_mask
from .zipformer.scaling import ScheduledFloat
from .zipformer.subsampling import Conv2dSubsampling
from .zipformer.zipformer import Zipformer2

logger = logging.getLogger(__name__)

WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin", "model.pt")


def _build_text_encoder(cfg: dict) -> nn.Module:
    from transformers import RobertaConfig, RobertaModel

    cfg = {k: v for k, v in cfg.items() if k != "model_type"}
    return RobertaModel(RobertaConfig(**cfg), add_pooling_layer=False)


class CaptionStewCLAP(nn.Module):
    """Contrastive audio-text model trained on CaptionStew.

    Main entry points:
      * :meth:`from_pretrained` - load a converted checkpoint (local dir or HF repo)
      * :meth:`encode_audio` - audio (paths / waveforms / fbank) -> (B, 1024) embeddings
      * :meth:`encode_text` - captions -> (B, 1024) embeddings
      * :meth:`audio_frame_features` - frame-level Zipformer representations (B, T, 512)
    """

    def __init__(self, config: Optional[ClapConfig] = None):
        super().__init__()
        self.config = config = config or ClapConfig()
        ac = config.audio_encoder

        # ---- audio tower (parameter names match the training checkpoints) ----
        self.encoder_embed = Conv2dSubsampling(
            in_channels=ac.feature_dim,
            out_channels=ac.encoder_dim[0],
            dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        )
        self.encoder = Zipformer2(
            output_downsampling_factor=ac.output_downsampling_factor,
            downsampling_factor=tuple(ac.downsampling_factor),
            encoder_dim=ac.encoder_dim,
            num_encoder_layers=ac.num_encoder_layers,
            encoder_unmasked_dim=ac.encoder_unmasked_dim,
            query_head_dim=ac.query_head_dim,
            pos_head_dim=ac.pos_head_dim,
            value_head_dim=ac.value_head_dim,
            num_heads=ac.num_heads,
            feedforward_dim=ac.feedforward_dim,
            cnn_module_kernel=ac.cnn_module_kernel,
            pos_dim=ac.pos_dim,
            dropout=ac.dropout,
            warmup_batches=ac.warmup_batches,
            causal=ac.causal,
            chunk_size=tuple(ac.chunk_size),
            left_context_frames=tuple(ac.left_context_frames),
        )
        self.audio_encoder_dim = max(ac.encoder_dim)

        # ---- text tower ----
        self.text_encoder = _build_text_encoder(config.text_encoder)
        self.text_encoder_dim = config.text_encoder["hidden_size"]

        # ---- projection heads ----
        d = config.shared_emb_dim
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_encoder_dim, d), nn.ReLU(), nn.Linear(d, d)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder_dim, d), nn.ReLU(), nn.Linear(d, d)
        )

        self.fbank = Fbank(sampling_rate=config.sampling_rate, num_mel_bins=config.num_mel_bins)
        self.tokenizer = None  # set by from_pretrained / load_tokenizer

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        device: Union[str, torch.device] = "cpu",
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> "CaptionStewCLAP":
        """Load a converted model from a local directory or a Hugging Face model repo.

        The directory must contain ``config.json`` and ``model.safetensors``
        (see ``scripts/convert_checkpoint.py``) and, ideally, the tokenizer files.
        """
        if os.path.isdir(name_or_path):
            model_dir = name_or_path
        else:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(
                name_or_path, revision=revision, cache_dir=cache_dir
            )
        config = ClapConfig.from_pretrained(model_dir)
        model = cls(config)
        weight_path = next(
            (os.path.join(model_dir, n) for n in WEIGHT_NAMES if os.path.exists(os.path.join(model_dir, n))),
            None,
        )
        if weight_path is None:
            raise FileNotFoundError(f"No weights ({', '.join(WEIGHT_NAMES)}) found in {model_dir}")
        model.load_state_dict(load_state_dict(weight_path), strict=True)
        model.load_tokenizer(model_dir)
        return model.to(device).eval()

    def load_tokenizer(self, model_dir: Optional[str] = None):
        """Load the tokenizer saved with the model (``<model_dir>/tokenizer``) or from the Hub."""
        from transformers import AutoTokenizer

        local = os.path.join(model_dir, "tokenizer") if model_dir else None
        if local and os.path.exists(os.path.join(local, "tokenizer_config.json")):
            self.tokenizer = AutoTokenizer.from_pretrained(local)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        return self.tokenizer

    def save_pretrained(self, save_directory: str):
        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(state, os.path.join(save_directory, "model.safetensors"), metadata={"format": "pt"})
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------ audio
    def compute_features(self, audios: Sequence[AudioInput]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Paths / 1-D waveforms (16 kHz) -> padded fbank ``(B, T, 80)`` and lengths ``(B,)``."""
        if isinstance(audios, str) or (
            isinstance(audios, (torch.Tensor, np.ndarray)) and audios.ndim == 1
        ):
            audios = [audios]
        return compute_fbank(list(audios), self.fbank, device=self.device)

    def _as_features(self, audio) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            isinstance(audio, tuple)
            and len(audio) == 2
            and isinstance(audio[0], torch.Tensor)
            and audio[0].dim() == 3
        ):
            x, x_lens = audio
        else:
            x, x_lens = self.compute_features(audio)
        dtype = self.audio_proj[0].weight.dtype
        return x.to(self.device, dtype), x_lens.to(self.device)

    def audio_frame_features(
        self, audio, x_lens: Optional[torch.Tensor] = None, all_layers: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Run the Zipformer encoder and return frame-level outputs.

        Args:
            audio: list of paths / waveforms, or a ``(features, feature_lens)`` tuple,
                or a feature tensor together with ``x_lens``.
            all_layers: also return the outputs of each of the 6 Zipformer stacks.

        Returns:
            dict with ``encoder_out`` (B, T', 512) at 25 Hz, ``encoder_out_lens`` (B,)
            and optionally ``layer_outputs`` (list of (B, T', C_i) tensors).
        """
        if x_lens is not None:
            audio = (audio, x_lens)
        x, x_lens = self._as_features(audio)
        x, x_lens = self.encoder_embed(x, x_lens)
        src_key_padding_mask = make_pad_mask(x_lens, max_len=x.size(1))
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        out, out_lens, x_full = self.encoder(x, x_lens, src_key_padding_mask)
        result = {"encoder_out": out.permute(1, 0, 2), "encoder_out_lens": out_lens}
        if all_layers:
            # x_full: (num_stacks, T', B, max_dim), zero-padded beyond each stack's dim
            dims = self.config.audio_encoder.encoder_dim
            result["layer_outputs"] = [
                x_full[i].permute(1, 0, 2)[..., : dims[i]] for i in range(len(dims))
            ]
        return result

    def encode_audio(
        self, audio, x_lens: Optional[torch.Tensor] = None, normalize: bool = True
    ) -> torch.Tensor:
        """Audio -> clip embeddings ``(B, shared_emb_dim)``.

        Frames are projected with ``audio_proj`` and then mean-pooled over valid frames.
        """
        out = self.audio_frame_features(audio, x_lens)
        feats = self.audio_proj(out["encoder_out"])
        pad = make_pad_mask(out["encoder_out_lens"], max_len=feats.size(1)).to(feats.device)
        feats = feats.masked_fill(pad.unsqueeze(-1), 0.0)
        counts = (~pad).sum(dim=1).clamp(min=1).unsqueeze(-1)
        emb = feats.sum(dim=1) / counts
        return F.normalize(emb, dim=-1) if normalize else emb

    # ------------------------------------------------------------------ text
    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
        preprocess: Optional[bool] = None,
    ) -> torch.Tensor:
        """Captions -> text embeddings ``(B, shared_emb_dim)`` (``<s>`` token + ``text_proj``)."""
        if isinstance(text, str):
            text = [text]
        if self.tokenizer is None:
            self.load_tokenizer()
        if preprocess is None:
            preprocess = self.config.lowercase_and_strip_punctuation
        if preprocess:
            text = [preprocess_caption(t) for t in text]
        enc = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.text_encoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask)[0]
        emb = self.text_proj(hidden[:, 0, :])
        return F.normalize(emb, dim=-1) if normalize else emb

    # ------------------------------------------------------------------ both
    def forward(self, audio, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(audio_embeds, text_embeds)``."""
        return self.encode_audio(audio), self.encode_text(text)

    @torch.no_grad()
    def similarity(self, audio, text: Union[str, List[str]]) -> torch.Tensor:
        """Cosine similarity matrix of shape ``(num_audio, num_text)``."""
        return self.encode_audio(audio) @ self.encode_text(text).T


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.get("state_dict", obj) if isinstance(obj, dict) else obj
