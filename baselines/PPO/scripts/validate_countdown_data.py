#!/usr/bin/env python3
"""Compare countdown parquet shard with JSON export to ensure schema consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_parquet_sample(path: Path, n: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.head(n)


def load_json_sample(path: Path, n: int) -> list[dict]:
    path = path.expanduser()
    records: list[dict] = []
    with path.open() as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse JSON file {path}: {err}") from err
        if isinstance(data, list):
            records = data[:n]
        else:
            raise ValueError("JSON root must be a list of objects")
    return records


def normalize_json_record(record: dict) -> dict:
    prompt = record.get("context")
    # Wrap context into expected chat format
    prompt_chat = [{"role": "user", "content": prompt}]
    reward_model = {
        "style": "rule",
        "ground_truth": {
            "numbers": record.get("numbers"),
            "target": record.get("target"),
            "solution": record.get("solution"),
        },
    }
    extra = {
        "index": record.get("id"),
        "split": record.get("split", "train"),
        "task_id": record.get("task_id"),
    }
    return {
        "target": record.get("target"),
        "nums": record.get("numbers"),
        "data_source": record.get("data_source", "countdown"),
        "prompt": prompt_chat,
        "ability": record.get("ability", "math"),
        "reward_model": reward_model,
        "extra_info": extra,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare countdown parquet vs JSON samples.")
    parser.add_argument("--parquet", required=True, type=Path, help="Path to countdown/train.parquet")
    parser.add_argument("--json", required=True, type=Path, help="Path to countdown.json")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows to inspect")
    args = parser.parse_args()

    parquet_df = load_parquet_sample(args.parquet, args.rows)
    json_records = load_json_sample(args.json, args.rows)

    norm_json = [normalize_json_record(rec) for rec in json_records]

    def to_serializable(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    for idx in range(args.rows):
        print(f"\n=== Record {idx} ===")
        if idx < len(parquet_df):
            parquet_row = {k: to_serializable(v) for k, v in parquet_df.iloc[idx].to_dict().items()}
            print("Parquet:", json.dumps(parquet_row, ensure_ascii=False, indent=2, default=to_serializable))
        if idx < len(norm_json):
            print("JSON   :", json.dumps(norm_json[idx], ensure_ascii=False, indent=2, default=to_serializable))


if __name__ == "__main__":
    main()

