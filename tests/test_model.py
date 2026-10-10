import math

import pytest
import torch

from gpt_from_scratch import GPT, GPTConfig
from gpt_from_scratch.model import CausalSelfAttention

TINY = GPTConfig(block_size=16, vocab_size=65, n_layer=2, n_head=2, n_embd=32)


def test_gpt2_124m_parameter_count() -> None:
    with torch.device("meta"):
        model = GPT(GPTConfig(vocab_size=50257))
    assert sum(p.numel() for p in model.parameters()) == 124_439_808


def test_output_shapes_and_initial_loss() -> None:
    torch.manual_seed(0)
    model = GPT(TINY)
    x = torch.randint(0, 65, (4, 16))
    logits, loss = model(x, torch.randint(0, 65, (4, 16)))
    assert logits.shape == (4, 16, 65)
    assert loss is not None
    assert abs(float(loss.detach()) - math.log(65)) < 0.3  # starts near uniform


def test_no_targets_means_no_loss() -> None:
    _, loss = GPT(TINY)(torch.zeros((1, 4), dtype=torch.long))
    assert loss is None


def test_weight_tying() -> None:
    model = GPT(TINY)
    assert model.transformer["wte"].weight is model.lm_head.weight


def test_sequence_longer_than_block_size_rejected() -> None:
    with pytest.raises(ValueError):
        GPT(TINY)(torch.zeros((1, 17), dtype=torch.long))


def test_attention_is_causal() -> None:
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    a = torch.randint(0, 65, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 65  # change only the future
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-5)
    assert not torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-5)


def test_manual_attention_matches_flash() -> None:
    torch.manual_seed(0)
    flash = CausalSelfAttention(TINY, flash=True)
    manual = CausalSelfAttention(TINY, flash=False)
    manual.load_state_dict(flash.state_dict())
    x = torch.randn(3, 10, TINY.n_embd)
    assert torch.allclose(flash(x), manual(x), atol=1e-5)


def test_optimizer_decays_only_matrices() -> None:
    opt = GPT(TINY).configure_optimizers(weight_decay=0.1, learning_rate=1e-3)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_generate_extends_and_respects_block_size() -> None:
    torch.manual_seed(0)
    model = GPT(TINY)
    g = torch.Generator().manual_seed(1)
    out = model.generate(torch.zeros((2, 1), dtype=torch.long), 40, top_k=5, generator=g)
    assert out.shape == (2, 41)  # longer than block_size: the context is cropped
    assert int(out.max()) < 65
    assert model.training  # restores the mode it was in
