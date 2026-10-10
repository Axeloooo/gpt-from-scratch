"""Lecture 9's training recipe: warmup + cosine LR, grad accumulation, clipping, weight decay."""

from __future__ import annotations

import argparse
import math

import torch

from gpt_from_scratch.data import CharTokenizer, DataLoaderLite, load_text, split_tokens
from gpt_from_scratch.model import GPT, GPTConfig


def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup, then cosine decay down to `min_lr`."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def train(
    model: GPT,
    loader: DataLoaderLite,
    max_steps: int,
    grad_accum_steps: int = 1,
    max_lr: float = 6e-4,
    min_lr_ratio: float = 0.1,
    warmup_steps: int = 10,
    weight_decay: float = 0.1,
    clip: float = 1.0,
    device: str = "cpu",
    verbose: bool = False,
) -> list[float]:
    """Train for `max_steps` optimizer steps; each averages `grad_accum_steps` micro-batches."""
    model.to(device).train()
    optimizer = model.configure_optimizers(weight_decay, max_lr)
    losses: list[float] = []
    for step in range(max_steps):
        optimizer.zero_grad()
        loss_accum = 0.0
        for _ in range(grad_accum_steps):
            x, y = loader.next_batch()
            _, loss = model(x.to(device), y.to(device))
            assert loss is not None
            loss = loss / grad_accum_steps  # so the summed gradient is the average
            loss_accum += float(loss.detach())
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        lr = get_lr(step, max_lr, max_lr * min_lr_ratio, warmup_steps, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        losses.append(loss_accum)
        if verbose:
            print(f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | norm {float(norm):.2f}")
    return losses


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train a GPT on Tiny Shakespeare (character level)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, _ = split_tokens(data)
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=tok.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(config)
    loader = DataLoaderLite(train_data, args.batch_size, args.block_size)
    losses = train(
        model,
        loader,
        args.steps,
        args.grad_accum,
        max_lr=args.lr,
        warmup_steps=max(1, args.steps // 20),
        device=args.device,
        verbose=True,
    )
    print(f"final loss {losses[-1]:.4f}")
    start = torch.zeros((1, 1), dtype=torch.long, device=args.device)
    out = model.generate(start, args.sample_tokens)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
