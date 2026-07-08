"""
cli.py
------
Command-line batch runner for the Automated Image Recognition Tool.

Usage:
    python cli.py path/to/image.jpg
    python cli.py path/to/folder_of_images/
    python cli.py img1.jpg img2.png img3.jpeg
"""

import argparse
import glob
import json
import os
import sys

from recognizer import recognizer

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def collect_filepaths(inputs):
    filepaths = []
    for item in inputs:
        if os.path.isdir(item):
            for ext in IMAGE_EXTS:
                filepaths.extend(glob.glob(os.path.join(item, f"*{ext}")))
        elif os.path.isfile(item):
            filepaths.append(item)
        else:
            print(f"Warning: '{item}' not found, skipping.", file=sys.stderr)
    return sorted(filepaths)


def main():
    parser = argparse.ArgumentParser(description="Run image recognition on files or a folder.")
    parser.add_argument("inputs", nargs="+", help="Image file(s) or a directory of images")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions per image")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of a table")
    args = parser.parse_args()

    recognizer.top_k = args.top_k

    filepaths = collect_filepaths(args.inputs)
    if not filepaths:
        print("No valid image files found.", file=sys.stderr)
        sys.exit(1)

    results = recognizer.predict_batch(filepaths)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    for r in results:
        print(f"\n=== {r.filename} ({r.inference_time_ms:.1f} ms) ===")
        for p in r.predictions:
            print(f"  {p.label:<30} {p.confidence * 100:5.1f}%")


if __name__ == "__main__":
    main()
