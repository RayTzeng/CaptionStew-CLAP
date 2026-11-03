"""Unit tests that run without downloading any checkpoint.

    pip install -e ".[dev]" && pytest tests
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from captionstew_clap import CaptionStewCLAP, ClapConfig, Fbank, preprocess_caption  # noqa: E402
from captionstew_clap.config import ROBERTA_BASE_CONFIG  # noqa: E402


def tiny_config() -> ClapConfig:
    text = dict(ROBERTA_BASE_CONFIG)
    text.update(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)
    cfg = ClapConfig(text_encoder=text, shared_emb_dim=16)
    a = cfg.audio_encoder
    a.num_encoder_layers = [1] * 6
    a.encoder_dim = [32, 32, 48, 64, 48, 32]
    a.feedforward_dim = [64, 64, 96, 128, 96, 64]
    a.encoder_unmasked_dim = [16, 16, 32, 32, 32, 16]
    a.num_heads = [2] * 6
    return cfg


class DummyTokenizer:
    """Whitespace tokenizer so the tests do not need the Hub."""

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        ids = [[0] + [3 + (hash(w) % 1000) for w in t.split()][: max_length - 2] + [2] for t in texts]
        n = max(len(i) for i in ids)
        input_ids = torch.tensor([i + [1] * (n - len(i)) for i in ids])
        mask = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i in ids])

        class Enc(dict):
            def to(self, device):
                return Enc({k: v.to(device) for k, v in self.items()})

            __getattr__ = dict.__getitem__

        return Enc(input_ids=input_ids, attention_mask=mask)

    def save_pretrained(self, path):
        pass


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = CaptionStewCLAP(tiny_config()).eval()
    m.tokenizer = DummyTokenizer()
    return m


def test_preprocess_caption():
    assert preprocess_caption("A Dog barks, loudly!") == "a dog barks loudly "
    assert preprocess_caption("Hello (world)") == "hello world "


def test_fbank_shapes():
    fb = Fbank()
    for n, frames in [(16000, 100), (16001, 100), (400, 3), (160000, 1000)]:
        assert fb(torch.zeros(n) + 0.01 * torch.randn(n)).shape == (1, frames, 80)


def test_fbank_matches_lhotse():
    lhotse = pytest.importorskip("lhotse")
    ref = lhotse.Fbank(lhotse.FbankConfig(num_mel_bins=80))
    fb = Fbank()
    rng = np.random.default_rng(0)
    for n in (16000, 23456, 555):
        w = (0.1 * rng.standard_normal(n)).astype(np.float32)
        np.testing.assert_allclose(fb(torch.from_numpy(w))[0].numpy(), ref.extract(w, 16000), atol=1e-3)


def test_encode_shapes_and_norm(model):
    wavs = [torch.randn(16000), torch.randn(24000)]
    with torch.inference_mode():
        a = model.encode_audio(wavs)
        t = model.encode_text(["a dog barks", "rain on a tin roof"])
        frames = model.audio_frame_features(wavs, all_layers=True)
    assert a.shape == (2, 16) and t.shape == (2, 16)
    torch.testing.assert_close(a.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(t.norm(dim=-1), torch.ones(2))
    # T fbank frames -> (T - 7) // 2 after the conv front-end -> halved again at the output
    assert frames["encoder_out_lens"].tolist() == [((t - 7) // 2 + 1) // 2 for t in (100, 150)]
    assert [o.shape[-1] for o in frames["layer_outputs"]] == model.config.audio_encoder.encoder_dim
    assert model.similarity(wavs, ["x", "y", "z"]).shape == (2, 3)


def test_input_forms_agree(model):
    w = torch.randn(16000)
    with torch.inference_mode():
        e1 = model.encode_audio(w)
        e2 = model.encode_audio(w.numpy())
        e3 = model.encode_audio(model.compute_features([w]))
    torch.testing.assert_close(e1, e2)
    torch.testing.assert_close(e1, e3)


def test_save_and_load_roundtrip(model, tmp_path):
    model.save_pretrained(str(tmp_path))
    with open(tmp_path / "config.json") as f:
        assert json.load(f)["model_type"] == "captionstew-clap"
    loaded = CaptionStewCLAP(ClapConfig.from_pretrained(str(tmp_path)))
    from captionstew_clap.model import load_state_dict

    loaded.load_state_dict(load_state_dict(str(tmp_path / "model.safetensors")), strict=True)
    loaded.eval().tokenizer = DummyTokenizer()
    w = torch.randn(16000)
    with torch.inference_mode():
        torch.testing.assert_close(loaded.encode_audio(w), model.encode_audio(w))


def test_training_config_conversion():
    flat = {"feature_dim": 80, "encoder_dim": [192, 256, 384, 512, 384, 256],
            "name": "zipformer-clap", "text_encoder_type": "roberta-base", "shared_emb_dim": 1024}
    cfg = ClapConfig.from_dict(flat)
    assert cfg.audio_encoder.encoder_dim[3] == 512 and cfg.shared_emb_dim == 1024
    with pytest.raises(ValueError):
        ClapConfig.from_dict({**flat, "pooling": "mhap"})


def test_retrieval_metrics_perfect():
    from evaluate_retrieval import retrieval_metrics

    emb = torch.eye(4)
    cap2clip = torch.tensor([0, 0, 1, 2, 3, 3])
    m = retrieval_metrics(emb, emb[cap2clip], cap2clip)
    assert m["text_to_audio"]["R@1"] == 100.0 and m["audio_to_text"]["R@1"] == 100.0
    assert m["audio_to_text"]["mAP@10"] == pytest.approx(100.0)
