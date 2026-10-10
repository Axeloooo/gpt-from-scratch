"""Data for lecture 7 (char-level Tiny Shakespeare) and lecture 9 (sequential token batches)."""

from __future__ import annotations

from pathlib import Path

import torch

DEFAULT_TEXT = Path(__file__).resolve().parents[2] / "data" / "input.txt"


class CharTokenizer:
    """One token per distinct character, as in lecture 7."""

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)


def load_text(path: str | Path = DEFAULT_TEXT) -> str:
    return Path(path).read_text(encoding="utf-8")


def split_tokens(
    data: torch.Tensor, train_fraction: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(train_fraction * len(data))
    return data[:n], data[n:]


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random (inputs, next-token targets) windows, as in lecture 7."""
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


class DataLoaderLite:
    """Walk through the tokens in order, wrapping around at the end (lecture 9)."""

    def __init__(self, tokens: torch.Tensor, batch_size: int, block_size: int) -> None:
        if len(tokens) < batch_size * block_size + 1:
            raise ValueError("not enough tokens for one batch")
        self.tokens = tokens
        self.batch_size = batch_size
        self.block_size = block_size
        self.position = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = self.batch_size, self.block_size
        buf = self.tokens[self.position : self.position + b * t + 1]
        x, y = buf[:-1].view(b, t), buf[1:].view(b, t)
        self.position += b * t
        if self.position + b * t + 1 > len(self.tokens):
            self.position = 0
        return x, y
