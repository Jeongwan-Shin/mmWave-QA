#!/usr/bin/env python3
"""
Split mm_actions.json by mode and create compressed one-line versions
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def split_and_compress(input_path: str, output_dir: str) -> None:
    """Split by mode and create compressed one-line versions."""
    print("="*80)
    print("Split and Compress mm_actions.json")
    print("="*80)
    
    # Load data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Total records: {len(data)}")
    
    # Split by mode
    train_data = []
    test_data = []
    
    for record in data:
        mode = record.get("info", {}).get("mode", "unknown")
        if mode == "train":
            train_data.append(record)
        elif mode == "test":
            test_data.append(record)
    
    print(f"Train records: {len(train_data)}")
    print(f"Test records: {len(test_data)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save original (formatted)
    with open(output_path / "mm_actions.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    with open(output_path / "mm_actions_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(output_path / "mm_actions_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # Save compressed (one-line) versions
    with open(output_path / "mm_actions_compressed.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    
    with open(output_path / "mm_actions_train_compressed.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, separators=(',', ':'))
    
    with open(output_path / "mm_actions_test_compressed.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, separators=(',', ':'))
    
    print("\nFiles created:")
    print(f"  Formatted (pretty):")
    print(f"    - {output_path}/mm_actions.json")
    print(f"    - {output_path}/mm_actions_train.json") 
    print(f"    - {output_path}/mm_actions_test.json")
    print(f"  Compressed (one-line):")
    print(f"    - {output_path}/mm_actions_compressed.json")
    print(f"    - {output_path}/mm_actions_train_compressed.json")
    print(f"    - {output_path}/mm_actions_test_compressed.json")
    
    # Show file sizes
    print(f"\nFile sizes:")
    for filename in ["mm_actions.json", "mm_actions_train.json", "mm_actions_test.json"]:
        formatted_size = (output_path / filename).stat().st_size
        compressed_size = (output_path / f"{filename.replace('.json', '')}_compressed.json").stat().st_size
        ratio = (1 - compressed_size / formatted_size) * 100
        print(f"  {filename}: {formatted_size:,} bytes")
        print(f"  {filename.replace('.json', '')}_compressed.json: {compressed_size:,} bytes ({ratio:.1f}% smaller)")


def main():
    parser = argparse.ArgumentParser(description="Split and compress mm_actions.json")
    parser.add_argument(
        "--input",
        type=str,
        default="/workspace/mmWave-QA/mmWaveQA_benchmark/mm_actions.json",
        help="Path to mm_actions.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/mmWave-QA/mmWaveQA_benchmark/split_compressed",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        return
    
    split_and_compress(args.input, args.output_dir)


if __name__ == "__main__":
    main()








