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
`devel` via PR. Feature PRs are squash-merged, so the PR title must be a valid conventional commit. Releases: open a PR from `devel` into `main` and merge it with a merge commit (not squash), so semantic-release sees every commit. `release.yaml` runs semantic-release on `main` with the built-in `GITHUB_TOKEN`, tags the version, updates `CHANGELOG.md` and merges `main` back into `devel`. Never commit to `main` directly.
