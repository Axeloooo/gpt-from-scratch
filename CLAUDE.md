# gpt-from-scratch

GPT from scratch (Karpathy Zero to Hero, lectures 7 and 9). PyTorch.

## Setup

```
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
npm install                       # commitlint, prettier, semantic-release
pre-commit install --install-hooks
```

## Layout

- `src/gpt_from_scratch/`: `model.py`, `data.py`, `train.py`
- `tests/`: pytest, shape, causality, parameter count and training checks
- Root: tooling (`.pre-commit-config.yaml`, `.github/`, `package.json`, commitlint, semantic-release, prettier)

## Commands

- Format: `black .`; lint: `ruff check .` (ruff's formatter is not used)
- Types: `mypy` (strict)
- Tests: `pytest -q`
- Other files: `npx prettier --write .`

## Commits (Conventional Commits, enforced by commitlint)

Types: feat, fix, docs, infra, refactor, test, chore (also perf, build, ci, revert).

Authorship: Claude is never the author or co-author of a commit. Commits carry the author's name
and email only, with no `Co-Authored-By` or `Claude-Session` trailers. The author and committer
on this repo are Axel. commitlint rejects messages that name Claude in a trailer.

## Branches and releases

`feature/short-description`, `fix/...`, `docs/...`, `infra/...`. Branch from `devel`; work lands on
`devel` via PR. Releases go from `devel` to `main` via a release PR. semantic-release runs on
`main`, tags the version and updates `CHANGELOG.md`. Never commit to `main` directly.
