# Contributing

## Setup

```bash
make init
```

Creates `.venv`, installs all dependencies, and sets up pre-commit hooks.

**Requirements:** Python 3.10+ and a Rust toolchain (`rustup`).

## Make Commands

Run `make help` for the list of available targets.

## Commit Messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/). A `commit-msg` hook rejects commits that don't match the format.

```
<type>[(<scope>)][!]: <description>
```

**Common types:** `feat`, `fix`, `perf`, `refactor`, `docs`, `chore`. See [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) for the full enforced list.

**Examples:**
```bash
git commit -m "feat: add batch decomposition API"
git commit -m "fix: handle degenerate Weyl face in recovery"
git commit -m "perf!: remove JAX dependency"   # breaking change
```

The `!` suffix or a `BREAKING CHANGE:` footer marks a breaking change, which appears in release notes regardless of type.

Only `feat`, `fix`, `perf`, and `merge` appear in the auto-generated changelog.

## Pre-commit Hooks

Hooks defined in [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) run on every commit. They enforce linting (`ruff`), formatting, conventional-commit messages, and basic safety checks. The same hooks run in CI on every PR.

## Release Flow

### 1. Tag and push

```bash
git tag v0.X.0
git push origin main
git push origin v0.X.0
```

The tag push triggers the **Release** workflow (`release.yml`):
- Builds wheels via `cibuildwheel`
- Generates a changelog from conventional commits
- Creates a **draft** GitHub Release with wheels attached

### 2. Review and publish

- Go to [Releases](https://github.com/evmckinney9/gulps/releases)
- Edit the draft: add release notes, verify wheels are attached
- Click **Publish release**

Publishing triggers the **Publish** workflow (`publish.yml`), which uploads the wheels to PyPI.
