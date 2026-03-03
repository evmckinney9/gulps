#!/usr/bin/env bash
# collect_backlog.sh — Walk git history and collect benchmark timing at each commit.
#
# Outputs: scripts/commit_history/wsl_desktop/backlog.csv  (appended, safe to re-run; skips already-collected commits)
#
# Usage:
#   ./scripts/commit_history/collect_backlog.sh              # run newest to oldest
#   ./scripts/commit_history/collect_backlog.sh --limit 20   # only process 20 commits
#   ./scripts/commit_history/collect_backlog.sh --start abc123  # start from a specific commit
#   ./scripts/commit_history/collect_backlog.sh --dry-run       # run HEAD only, don't write to CSV
#
# When the benchmark fails (API changed, import error, etc.), the script stops
# and prints the full error. Fix benchmark_task.py, then re-run with --start.
#
# Uses `git worktree` so your working directory stays untouched.
#
# Caveats:
#   - Results are machine-specific and only comparable within a single machine.
#   - Assumes the machine is otherwise idle. Background load inflates timings,
#     which matters most at recent commits where decomposition is fast (~10ms).
#   - Older commits have higher numerical error rates (solver non-convergence).
#     Failed decompositions still contribute their timing to the median — they
#     exhaust all solver attempts before failing, so they're genuinely slower.
#     The progress line in benchmark_task.py shows `errors=N` when this occurs.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
BENCHMARK_SCRIPT="$REPO_ROOT/scripts/commit_history/benchmark_task.py"
OUTPUT_DIR="$REPO_ROOT/scripts/commit_history/wsl_desktop"
OUTPUT_CSV="$OUTPUT_DIR/backlog.csv"
WORKTREE_DIR="$REPO_ROOT/.bench_worktree"

# --- known-bad commits (library bugs, not benchmark-script issues) ---
BAD_COMMITS=(
    "74a88a6bdf3bd12a99231df31c335fef54e82952"  # try_solve() got multiple values for argument 'step'
    "f24f71b608a31a1af3b487b9102a2767f8066efc"  # circular import: isa.py <-> coverage.py
    "a08765ffe4ec3e79cbc0227679745acee2129e6c"  # warmup fails: segment synthesis residual too high
    "c41746343d6f6192b76a3fee66c1abef5a6dc88f"  # warmup fails: ISAInvariants has no attribute 'identity' (commit message says "broken")
    "06368020af00d1464ee46609567d39c32d6d32c7"  # warmup fails: Segment synthesis did not converge (commit message says "broke again")
    "01dff767621d0fc77dc1b7929387212da1810414"  # warmup fails: Segment synthesis did not converge (commit message says "broke numerics?")
)

# --- parse args ---
LIMIT=0
START_COMMIT=""
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        --start) START_COMMIT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; LIMIT=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# --- CSV header ---
if [[ ! -f "$OUTPUT_CSV" ]]; then
    echo "commit,date,subject,status,isa1_median,isa2_median,isa3_median" > "$OUTPUT_CSV"
fi


# --- oldest commit to benchmark (don't go past this) ---
OLDEST_COMMIT="1e7924fc4f2d2d18f46c7c93c8a3811747879d59"

# --- collect commits oldest-first ---
COMMITS=$(git log --format="%H" --first-parent --reverse "${OLDEST_COMMIT}^..HEAD")

if [[ -n "$START_COMMIT" ]]; then
    FULL_START=$(git rev-parse "$START_COMMIT")
    COMMITS=$(echo "$COMMITS" | sed -n "/$FULL_START/,\$p")
fi

if [[ "$LIMIT" -gt 0 ]]; then
    COMMITS=$(echo "$COMMITS" | tail -n "$LIMIT")
fi

TOTAL=$(echo "$COMMITS" | wc -l)
echo "=== Collecting benchmarks for $TOTAL commits (oldest first) ==="
echo "=== Machine: $(hostname), 1000 unitaries x 3 ISAs per commit ==="

# --- already collected set (for skip logic) ---
COLLECTED=$(cut -d',' -f1 "$OUTPUT_CSV" | tail -n +2)

cleanup_worktree() {
    if [[ -d "$WORKTREE_DIR" ]]; then
        git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
    fi
}
trap cleanup_worktree EXIT

IDX=0
for COMMIT in $COMMITS; do
    IDX=$((IDX + 1))
    SHORT=$(git log -1 --format="%h" "$COMMIT")
    DATE=$(git log -1 --format="%aI" "$COMMIT")
    SUBJECT=$(git log -1 --format="%s" "$COMMIT" | tr ',' ';')

    if ! $DRY_RUN && echo "$COLLECTED" | grep -q "^${COMMIT}$"; then
        echo "[$IDX/$TOTAL] $SHORT — skip (already collected)"
        continue
    fi

    if ! $DRY_RUN; then
        # Skip known-bad commits
        for BAD in "${BAD_COMMITS[@]}"; do
            if [[ "$COMMIT" == "$BAD" ]]; then
                echo "[$IDX/$TOTAL] $SHORT — skip (known-bad commit)"
                continue 2
            fi
        done

        if [[ "$SUBJECT" == wip* ]]; then
            echo "[$IDX/$TOTAL] $SHORT — skip (wip commit)"
            continue
        fi

        # Skip commits that don't touch any .py files in the package
        PY_CHANGED=$(git diff-tree --no-commit-id --name-only -r "$COMMIT" 2>/dev/null | grep -c '^src/gulps/.*\.py$' || true)
        if [[ "$PY_CHANGED" -eq 0 ]]; then
            echo "[$IDX/$TOTAL] $SHORT — skip (no src/gulps/ .py changes)"
            continue
        fi
    fi

    echo -n "[$IDX/$TOTAL] $SHORT $SUBJECT ... "

    IS_HEAD=false
    if [[ "$COMMIT" == "$(git rev-parse HEAD)" ]]; then
        IS_HEAD=true
    fi

    if $IS_HEAD; then
        # HEAD: benchmark in-place — no worktree, no reinstall needed
        ERRLOG="/tmp/_bench_head_err.log"
        BENCH_SCRIPT="$BENCHMARK_SCRIPT"
    else
        # Historical commit: check out via worktree
        cleanup_worktree
        git worktree add --detach "$WORKTREE_DIR" "$COMMIT" 2>/dev/null

        cp "$BENCHMARK_SCRIPT" "$WORKTREE_DIR/benchmark_task.py"

        ERRLOG="$WORKTREE_DIR/_bench_err.log"
        BENCH_SCRIPT="$WORKTREE_DIR/benchmark_task.py"

        pushd "$WORKTREE_DIR" > /dev/null

        if [[ ! -f "pyproject.toml" ]] && [[ ! -f "setup.py" ]]; then
            echo ""
            echo "STOPPED at $SHORT: no pyproject.toml or setup.py"
            cleanup_worktree
            exit 1
        fi

        # If this commit uses the monodromy fork, uninstall first so pip fetches the correct version
        if grep -q "monodromy" pyproject.toml 2>/dev/null; then
            "$REPO_ROOT/.venv/bin/pip" uninstall monodromy -y --quiet 2>/dev/null || true
        fi

        # Install — show errors if it fails
        if ! "$REPO_ROOT/.venv/bin/pip" install -e . --quiet 2>"$ERRLOG"; then
            echo ""
            echo "STOPPED at $SHORT: pip install failed"
            echo "--- error log ---"
            cat "$ERRLOG"
            popd > /dev/null
            cleanup_worktree
            exit 1
        fi
    fi

    # Run benchmark — stdout=JSON, stderr=progress+errors (tee to terminal and log)
    RESULT=""
    BENCH_ARGS=""
    if $DRY_RUN; then BENCH_ARGS="-n 10"; fi
    if RESULT=$("$REPO_ROOT/.venv/bin/python" "$BENCH_SCRIPT" $BENCH_ARGS 2> >(tee "$ERRLOG" >&2)); then
        CSV_VALS=$("$REPO_ROOT/.venv/bin/python" -c "
import json, sys
d = json.loads(sys.argv[1])
print(','.join(str(d.get(k, '')) for k in ['isa1','isa2','isa3']))
" "$RESULT")
        if $DRY_RUN; then
            echo "DRY RUN (not written to CSV)"
            echo "  result: $CSV_VALS"
        else
            echo "$COMMIT,$DATE,$SUBJECT,OK,$CSV_VALS" >> "$OUTPUT_CSV"
        fi
        echo "done ($(echo "$CSV_VALS" | tr ',' ' / '))"
    else
        echo ""
        echo "STOPPED at $SHORT: benchmark_task.py failed"
        echo "--- error log ---"
        cat "$ERRLOG"
        if ! $IS_HEAD; then popd > /dev/null; fi
        cleanup_worktree
        exit 1
    fi

    if ! $IS_HEAD; then popd > /dev/null; fi
done

# Restore the editable install from the repo root so the venv points at HEAD,
# not the (now-deleted) worktree directory.
echo ""
echo "Restoring editable install from HEAD..."
"$REPO_ROOT/.venv/bin/pip" install -e "$REPO_ROOT" --quiet

echo ""
echo "=== Done. Results in $OUTPUT_CSV ==="
echo "Total collected: $(tail -n +2 "$OUTPUT_CSV" | wc -l)"
