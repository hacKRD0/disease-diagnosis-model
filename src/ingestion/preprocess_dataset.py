"""Pre-process the raw symptom–disease CSVs.
Reads the raw train / validation CSV files found in `data/raw/`,
converts the numeric label IDs to their textual disease names using
`label_mapping.json`, and writes the processed CSVs to `data/processed/`.

Usage (from repo root):

    python -m src.ingestion.preprocess_dataset \
        --train_csv data/raw/symptom-disease-train-dataset.csv \
        --val_csv   data/raw/symptom-disease-test-dataset.csv \
        --mapping_file data/label_mapping.json

You may override the output directory via `--output_dir`.
"""

import argparse
import json
import os
import sys
from typing import Dict

try:
    import pandas as pd  # type: ignore
except ImportError as e:
    sys.stderr.write("pandas is required for preprocessing; install with `pip install pandas`\n")
    raise


def load_mapping(mapping_path: str) -> Dict[int, str]:
    """Load the label-ID->name mapping from JSON and invert it to ID->name."""
    with open(mapping_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # The original mapping file is assumed to be {name: id}. Invert it.
    return {int(v): k for k, v in raw.items()}


def process_csv(csv_path: str, mapping: Dict[int, str], output_dir: str) -> None:
    """Read a CSV, map its `label` column, and write the processed CSV."""
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"Expected a 'label' column in {csv_path} but found {list(df.columns)}")

    try:
        df["label"] = df["label"].map(lambda x: mapping[int(x)])
    except KeyError as missing:
        raise KeyError(f"Label ID {missing} in {csv_path} is not present in the mapping file") from None

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(csv_path))
    df.to_csv(output_path, index=False)
    print(f"💾 Saved processed CSV to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert numeric labels to text labels in dataset CSVs.")
    parser.add_argument("--train_csv", required=True, help="Path to raw training CSV file")
    parser.add_argument("--val_csv", required=True, help="Path to raw validation CSV file")
    parser.add_argument("--mapping_file", required=True, help="Path to label-mapping JSON file")
    parser.add_argument("--output_dir", default="data/processed", help="Directory to save processed CSVs")
    args = parser.parse_args()

    mapping = load_mapping(args.mapping_file)

    for csv_path in (args.train_csv, args.val_csv):
        process_csv(csv_path, mapping, args.output_dir)


if __name__ == "__main__":
    main()
