#!/usr/bin/env bash
# gay_ts_flox.sh - run gay peer with flox environment
# Usage: ./gay_ts_flox.sh [seed]

SEED=${1:-69}

# Check if flox is available
if command -v flox &> /dev/null; then
    echo "using flox environment"
    flox activate -- python3 "$(dirname "$0")/gay_ts.py" "$SEED"
else
    # Fallback to system python
    python3 "$(dirname "$0")/gay_ts.py" "$SEED"
fi
