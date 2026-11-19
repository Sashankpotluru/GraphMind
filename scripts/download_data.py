"""Run from the project root with python -m scripts.download_data."""

import argparse
from pathlib import Path

from benchmarks.sources import fetch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    fetch(parser.parse_args().cache)
