"""AirPhyNet preprocessing entry point — delegates to the shared pipeline."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocess import preprocess

if __name__ == "__main__":
    raw = os.path.join(os.path.dirname(__file__), "data", "aqi_all.csv")
    out = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    preprocess(raw, out)
