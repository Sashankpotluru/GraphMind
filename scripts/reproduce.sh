#!/bin/sh
# Recompute retrieval, timings and the matched reader into a new verified release.
set -eu
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: scripts/reproduce.sh NEW_RELEASE_DIR [CACHE_DIR]" >&2
    exit 2
fi
output=$1
cache=${2:-data/cache}
if [ -e "$output" ] || [ -L "$output" ]; then
    echo "Output exists. Choose a new release directory." >&2
    exit 2
fi
parent=$(dirname "$output")
mkdir -p "$parent"
work=$(mktemp -d "$parent/.graphmind-observations-XXXXXXXX")
echo "Private reproduction observations: $work"
python -m scripts.download_data --cache "$cache"
python -m benchmarks.run_answers --cache "$cache" --download-only
python -m benchmarks.run_retrieval --cache "$cache" --output "$work/retrieval"
python -m benchmarks.run_answers --cache "$cache" --retrieval "$work/retrieval" --output "$work/answers"
python -m benchmarks.release --observations "$work" --output "$output"
echo "Verified reproduction: $output"
