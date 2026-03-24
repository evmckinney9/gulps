# Contributing

## Setup

```bash
make init
```

This creates a `.venv`, installs all dependencies (including the Rust extension), and sets up pre-commit hooks for both the `pre-commit` and `commit-msg` stages.

**Requirements:** Python 3.10+ and a Rust toolchain (`rustup`).

## Make Commands

| Command         | Description                                                        |
| --------------- | ------------------------------------------------------------------ |
| **`init`**      | Create venv, install deps, set up pre-commit hooks. (Removes existing `.venv/`) |
| `upgrade`       | Upgrade all packages to latest versions.                           |
| `clean`         | Remove temporary files and build artifacts.                        |
| `test`          | Run pytest.                                                        |
| `format`        | Run all pre-commit hooks on all files.                             |
| **`precommit`** | Run tests, then format.                                            |

## Commit Messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/). A `commit-msg` hook enforces the format — commits that don't match will be rejected.

```
<type>[(<scope>)][!]: <description>
```

**Types:** `feat`, `fix`, `perf`, `refactor`, `style`, `test`, `build`, `ops`, `docs`, `chore`, `merge`, `revert`

**Examples:**
```bash
git commit -m "feat: add batch decomposition API"
git commit -m "fix: handle degenerate Weyl face in recovery"
git commit -m "perf!: remove JAX dependency"   # breaking change
```

The `!` suffix or a `BREAKING CHANGE:` footer marks a breaking change, which appears in release notes regardless of type.

Only `feat`, `fix`, `perf`, and `merge` appear in the auto-generated changelog.

## Pre-commit Hooks

Hooks run in order with `fail_fast: true` (stops on first failure):

1. **conventional-commits** — validates commit message format (`commit-msg` stage)
2. **ruff-check** — lint with auto-fix
3. **ruff-format** — code formatting
4. **trailing-whitespace / end-of-file-fixer** — whitespace normalization
5. **check-merge-conflict / detect-private-key / check-added-large-files** — safety checks
6. **name-tests-test** — enforces `test_*.py` naming
7. **enforce-notebook-naming** — enforces notebook naming conventions

## Release Flow

### 1. Tag and push

```bash
git tag v0.X.0
git push origin main
git push origin v0.X.0
```

The tag push triggers the **Release** workflow (`release.yml`):
- Builds wheels for Linux (x86_64), macOS (ARM64), and Windows using `cibuildwheel`
- Generates a changelog from conventional commits
- Creates a **draft** GitHub Release with wheels attached

### 2. Review and publish

- Go to [Releases](https://github.com/evmckinney9/gulps/releases)
- Edit the draft: add release notes, verify wheels are attached
- Click **Publish release**

Publishing triggers the **Publish** workflow (`publish.yml`):
- Downloads wheels from the GitHub Release
- Uploads to PyPI

### CI

Every push to `main` and every PR triggers the **CI** workflow (`ci.yml`):
- **format** job: runs pre-commit hooks (ruff, whitespace, etc.)
- **tests** job: installs the package with Rust extension, runs pytest

## Project Structure

```
crates/              # Rust extension (PyO3)
  src/
    ...
    lib.rs           # Python bindings
  Cargo.toml
  Cargo.lock         # pinned — checked into git

src/
  gulps/             # Python package
  tests/             # Pytests
  notebooks/         # Jupyter notebooks

.github/
  workflows/
    ci.yml           # lint + test on push/PR
    release.yml      # build wheels + draft release on tag
    publish.yml      # upload to PyPI on release publish
  ISSUE_TEMPLATE/
```
