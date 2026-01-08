#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json
from pathlib import Path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--labels", type=str, required=True)
    ap.add_argument("--out_labels", type=str, default="")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    labels = Path(args.labels)
    out_labels = Path(args.out_labels) if args.out_labels else labels.with_name(
        "labels.cleaned.jsonl")

    kept = 0
    missing = 0

    with labels.open("r", encoding="utf-8") as fin, out_labels.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                missing += 1
                continue

            rel = Path(obj.get("path", ""))
            p = data_dir / rel
            if p.exists():
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
            else:
                missing += 1

    print(f"written: {out_labels}")
    print(f"kept={kept} missing={missing}")


if __name__ == "__main__":
    main()
