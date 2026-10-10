import pytest
import torch

from gpt_from_scratch import GPT, CharTokenizer, DataLoaderLite, GPTConfig, get_batch, get_lr, train
from gpt_from_scratch.data import split_tokens
from gpt_from_scratch.train import main

TEXT = "to be, or not to be, that is the question. " * 200
CFG = GPTConfig(block_size=16, vocab_size=0, n_layer=2, n_head=2, n_embd=32)


def tokens() -> tuple[CharTokenizer, torch.Tensor]:
    tok = CharTokenizer(TEXT)
    return tok, torch.tensor(tok.encode(TEXT))


def test_char_tokenizer_round_trip() -> None:
    tok, ids = tokens()
    assert tok.decode(ids.tolist()) == TEXT
    assert tok.vocab_size == len(set(TEXT))


def test_get_batch_targets_are_shifted() -> None:
    _, data = tokens()
    x, y = get_batch(data, 4, 8, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_split_tokens() -> None:
    _, data = tokens()
    a, b = split_tokens(data)
    assert len(a) + len(b) == len(data) and len(a) == int(0.9 * len(data))


def test_dataloader_walks_and_wraps() -> None:
    data = torch.arange(100)
    loader = DataLoaderLite(data, batch_size=2, block_size=5)
    x, y = loader.next_batch()
    assert x.tolist()[0] == [0, 1, 2, 3, 4] and y.tolist()[0] == [1, 2, 3, 4, 5]
    assert loader.next_batch()[0].tolist()[0] == [10, 11, 12, 13, 14]
    for _ in range(20):
        loader.next_batch()
    assert 0 <= loader.position < 100
    with pytest.raises(ValueError):
        DataLoaderLite(torch.arange(5), 2, 5)


def test_lr_schedule() -> None:
    kw = {"max_lr": 1.0, "min_lr": 0.1, "warmup_steps": 10, "max_steps": 110}
    assert get_lr(0, **kw) == pytest.approx(0.1)  # type: ignore[arg-type]
    assert get_lr(9, **kw) == pytest.approx(1.0)  # type: ignore[arg-type]
    assert get_lr(60, **kw) == pytest.approx(0.55)  # type: ignore[arg-type]
    assert get_lr(110, **kw) == pytest.approx(0.1)  # type: ignore[arg-type]
    assert get_lr(10_000, **kw) == pytest.approx(0.1)  # type: ignore[arg-type]


def test_training_reduces_loss() -> None:
    torch.manual_seed(0)
    tok, data = tokens()
    model = GPT(GPTConfig(16, tok.vocab_size, 2, 2, 32))
    loader = DataLoaderLite(data, 8, 16)
    losses = train(model, loader, max_steps=60, max_lr=3e-3, warmup_steps=5)
    assert losses[-1] < losses[0] - 1.0


def test_grad_accumulation_matches_big_batch() -> None:
    _, data = tokens()
    cfg = GPTConfig(16, len(set(TEXT)), 1, 2, 16)

    def grads(batch: int, accum: int) -> list[torch.Tensor]:
        torch.manual_seed(0)
        model = GPT(cfg)
        loader = DataLoaderLite(data, batch, 16)
        train(model, loader, max_steps=1, grad_accum_steps=accum, clip=1e9, max_lr=0.0)
        return [p.grad.clone() for p in model.parameters() if p.grad is not None]

    big, accumulated = grads(4, 1), grads(2, 2)
    for a, b in zip(big, accumulated, strict=True):
        assert torch.allclose(a, b, atol=1e-5)


def test_cli_runs(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "--steps",
            "3",
            "--batch-size",
            "2",
            "--block-size",
            "8",
            "--n-layer",
            "1",
            "--n-head",
            "2",
            "--n-embd",
            "16",
            "--sample-tokens",
            "10",
        ]
    )
    assert "final loss" in capsys.readouterr().out
