# gpt-from-scratch

A GPT built from scratch in PyTorch, from a character-level transformer on Tiny Shakespeare up to
the GPT-2 124M architecture and training recipe. It follows two lectures from Andrej Karpathy's
Neural Networks: Zero to Hero series:

- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)

## Install

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Where each part lives

| Lecture step                                               | Code                                      |
| ---------------------------------------------------------- | ----------------------------------------- |
| Character tokenizer, train/val split, random batches       | `CharTokenizer`, `get_batch` in `data.py` |
| Masked self-attention, written out (scores, mask, softmax) | `CausalSelfAttention(flash=False)`        |
| Fused attention (`scaled_dot_product_attention`)           | `CausalSelfAttention(flash=True)`         |
| MLP, residual pre-norm blocks, GPT-2 architecture          | `MLP`, `Block`, `GPT` in `model.py`       |
| Weight tying, scaled residual init                         | `GPT.__init__`, `GPT._init_weights`       |
| AdamW with decay only on matrices                          | `GPT.configure_optimizers`                |
| Sampling with temperature and top-k                        | `GPT.generate`                            |
| Warmup + cosine LR, grad accumulation, clipping            | `get_lr`, `train` in `train.py`           |
| Sequential data loader                                     | `DataLoaderLite` in `data.py`             |

`GPTConfig()` defaults to GPT-2 124M (12 layers, 12 heads, 768 dims, vocabulary padded to 50304).
A test builds it on the `meta` device and checks the parameter count is exactly 124,439,808, the
same as Hugging Face's `gpt2`.

## Run

```
python -m gpt_from_scratch --steps 2000      # trains a small GPT on data/input.txt
```

The defaults are small (4 layers, 128 dims) so it runs on a CPU in seconds and prints a sample of
generated text. For a better model raise `--steps`, `--n-layer`, `--n-embd` and `--block-size`;
`--grad-accum` simulates a larger batch and `--device cuda` uses a GPU.

## Not included

The GPT-2 124M pretraining run itself (about 10B tokens of FineWeb-Edu), the HellaSwag evaluation,
loading Hugging Face weights, `torch.compile` and multi-GPU DDP are not part of this repository.
They need GPUs and hours of compute that this code was not run with. The model and training loop
are the same ones the lecture scales up, so the plumbing is here to do that run on your hardware.

## Tests

```
pytest -q
```

They check shapes, the initial loss being near `ln(vocab)`, causality (future tokens never change
earlier logits), that the written-out attention matches the fused one, weight tying, the exact
124M parameter count, optimizer groups, the LR schedule, that gradient accumulation matches one big
batch, and that training lowers loss.

## Development

See `CLAUDE.md`. Commits follow Conventional Commits; work lands on `devel` through pull requests.

## License

MIT
