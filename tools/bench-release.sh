#!/usr/bin/env bash
# Time this checkout against a release with `make bench`: tools/bench-release.sh [TAG]
# The default TAG is the last release before HEAD. The comparison is saved to build/bench-release.txt.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
base=${1:-$(git describe --tags --abbrev=0 HEAD^)}
mkdir -p build
uv pip install --python .venv/bin/python --no-deps --quiet "git+file://$PWD@$base"
make bench -o rebuild BENCH_ARGS="--benchmark-save=$base"
make bench BENCH_ARGS="--benchmark-save=HEAD --benchmark-compare --benchmark-sort=name --benchmark-columns=median,iqr,rounds" | tee build/bench-release.txt
