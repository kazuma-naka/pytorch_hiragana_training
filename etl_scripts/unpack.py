#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# unpack.py

from __future__ import annotations

import codecs
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List
import datetime
import json

import bitstring
from PIL import Image
import jaconv


def T56(c: int) -> str:
    t56s = '0123456789[#@:>? ABCDEFGHI&.](<  JKLMNOPQR-$*);\'|/STUVWXYZ ,%="!'
    return t56s[c]


class CO59_to_unicode:
    def __init__(self, euc_co59_file: str = "euc_co59.dat"):
        with codecs.open(euc_co59_file, "r", "euc-jp") as f:
            co59t = f.read()
        co59l = co59t.split()
        self.conv: Dict[tuple[int, int], str] = {}
        for c in co59l:
            ch = c.split(":")
            co = ch[1].split(",")
            co59c = (int(co[0]), int(co[1]))
            self.conv[co59c] = ch[0]

    def __call__(self, co59: tuple[int, int]) -> str:
        return self.conv[co59]


co59_to_unicode = CO59_to_unicode("euc_co59.dat")


@dataclass
class RecordType:
    length_in_octets: int
    fields: Dict[str, str]
    converters: Dict[str, Callable[[Dict[str, Any]], Any]]

    def read(self, filename: Path, skip_first: bool = False) -> List[Dict[str, Any]]:
        bs = ",".join(self.fields.values())
        keys = [k for k, v in self.fields.items() if "pad" not in v]

        records: List[Dict[str, Any]] = []
        with open(filename, "rb") as f:
            while True:
                chunk = f.read(self.length_in_octets)
                if not chunk:
                    break
                if len(chunk) != self.length_in_octets:
                    print("Warning: Incomplete record detected, skipping.")
                    continue
                if skip_first:
                    skip_first = False
                    continue

                s = bitstring.ConstBitStream(bytes=chunk)
                record = dict(zip(keys, s.unpack(bs)))

                # converters (order matters; dict preserves insertion order in Python 3.7+)
                for k, v in self.converters.items():
                    try:
                        record[k] = v(record)
                    except Exception as e:
                        print(
                            f"Warning: Error occurred in converting {k}: {e}")
                        # keep record, but skip this field
                        continue

                records.append(record)

        return records


# ---------------- Record Types ----------------

M_Type = RecordType(
    2052,
    {
        "Data Number": "uint:16",
        "Character Code": "bytes:2",
        "Serial Sheet Number": "uint:16",
        "JIS Code": "hex:8",
        "EBCDIC Code": "hex:8",
        "Evaluation of Individual Character Image": "uint:8",
        "Evaluation of Character Group": "uint:8",
        "Male-Female Code": "uint:8",
        "Age of Writer": "uint:8",
        "Serial Data Number": "uint:32",
        "Industry Classification Code": "uint:16",
        "Occupation Classification Code": "uint:16",
        "Sheet Gatherring Date": "uint:16",
        "Scanning Date": "uint:16",
        "Sample Position Y on Sheet": "uint:8",
        "Sample Position X on Sheet": "uint:8",
        "Minimum Scanned Level": "uint:8",
        "Maximum Scanned Level": "uint:8",
        "Undefined1": "pad:16",
        "Undefined2": "pad:16",
        "Image Data": "bytes:2016",
        "Undefined3": "pad:32",
    },
    {
        "char": lambda x: jaconv.h2z(bytes.fromhex(x["JIS Code"]).decode("shift_jis"), digit=False, ascii=False)
        .replace("ィ", "ヰ")
        .replace("ェ", "ヱ"),
        "unicode": lambda x: hex(ord(x["char"])),
        "image": lambda x: Image.eval(
            Image.frombytes(
                "F", (64, 63), x["Image Data"], "bit", 4).convert("L"),
            lambda p: p * 16,
        ),
    },
)

K_Type = RecordType(
    2745,
    {
        "Serial Data Number": "uint:36",
        "Mark of Style": "uint:6",
        "Spaces": "pad:30",
        "Contents": "bits:36",
        "Style": "bits:36",
        "Zeros": "pad:24",
        "CO-59 Code": "bits:12",
        "(undefined)": "pad:180",
        "Image Data": "bytes:2700",
    },
    {
        "co-59_code": lambda x: tuple([b.uint for b in x["CO-59 Code"].cut(6)]),
        "char": lambda x: co59_to_unicode(x["co-59_code"]),
        "unicode": lambda x: hex(ord(x["char"])),
        "image": lambda x: Image.eval(
            Image.frombytes(
                "F", (60, 60), x["Image Data"], "bit", 6).convert("L"),
            lambda p: p * 4,
        ),
        "mark_of_style": lambda x: T56(x["Mark of Style"]),
        "contents": lambda x: "".join([T56(b.uint) for b in x["Contents"].cut(6)]),
        "style": lambda x: "".join([T56(b.uint) for b in x["Style"].cut(6)]),
    },
)

# ★ ETL3/4/5 の C_Type を修正
C_Type = RecordType(
    2952,
    {
        "Serial Data Number": "uint:36",
        "Serial Sheet Number": "uint:36",
        "JIS Code": "hex:8",
        "(pad1)": "pad:28",
        "EBCDIC Code": "hex:8",
        "(pad2)": "pad:28",
        "4 Character Code": "bits:24",
        "Spaces": "bits:12",
        "Evaluation of Individual Character Image": "uint:36",
        "Evaluation of Character Group": "uint:36",
        "Sample Position Y on Sheet": "uint:36",
        "Sample Position X on Sheet": "uint:36",
        "Male-Female Code": "uint:36",
        "Age of Writer": "uint:36",
        "Industry Classification Code": "uint:36",
        "Occupation Classification Code": "uint:36",
        "Sheet Gatherring Date": "uint:36",
        "Scanning Date": "uint:36",
        "Number of X-Axis Sampling Points": "uint:36",
        "Number of Y-Axis Sampling Points": "uint:36",
        "Number of Levels of Pixel": "uint:36",
        "Magnification of Scanning Lenz": "uint:36",
        "Serial Data Number (old)": "uint:36",
        "(undefined)": "pad:1008",
        "Image Data": "bytes:2736",
    },
    {
        "fourcc": lambda x: "".join([T56(b.uint) for b in x["4 Character Code"].cut(6)]),
        "spaces": lambda x: "".join([T56(b.uint) for b in x["Spaces"].cut(6)]),
        "image": lambda x: Image.eval(
            Image.frombytes(
                "F", (72, 76), x["Image Data"], "bit", 4).convert("L"),
            lambda p: p * 16,
        ),
        "_char": lambda x: bytes.fromhex(x["JIS Code"]).decode("shift_jis"),
        "char": lambda x: (
            # H: Hiragana（元がカタカナ等の場合もあるので hira 化）
            jaconv.kata2hira(jaconv.han2zen(x["_char"])).replace(
                "ぃ", "ゐ").replace("ぇ", "ゑ")
            if x["fourcc"] and x["fourcc"][0] == "H"
            else
            # K: Katakana（全角化）
            jaconv.han2zen(x["_char"]).replace("ィ", "ヰ").replace("ェ", "ヱ")
            if x["fourcc"] and x["fourcc"][0] == "K"
            else x["_char"]
        ),
        "unicode": lambda x: hex(ord(x["char"])),
    },
)

G8_Type = RecordType(
    8199,
    {
        "Serial Sheet Number": "uint:16",
        "JIS Kanji Code": "hex:16",
        "JIS Typical Reading": "bytes:8",
        "Serial Data Number": "uint:32",
        "Evaluation of Individual Character Image": "uint:8",
        "Evaluation of Character Group": "uint:8",
        "Male-Female Code": "uint:8",
        "Age of Writer": "uint:8",
        "Industry Classification Code": "uint:16",
        "Occupation Classification Code": "uint:16",
        "Sheet Gatherring Date": "uint:16",
        "Scanning Date": "uint:16",
        "Sample Position X on Sheet": "uint:8",
        "Sample Position Y on Sheet": "uint:8",
        "(undefined)": "pad:240",
        "Image Data": "bytes:8128",
        "(uncertain)": "pad:88",
    },
    {
        "JIS_kanji_code": lambda x: "1b2442" + x["JIS Kanji Code"] + "1b2842",
        "JIS_typical_reading": lambda x: x["JIS Typical Reading"].decode("ascii"),
        "char": lambda x: bytes.fromhex(x["JIS_kanji_code"]).decode("iso2022_jp"),
        "unicode": lambda x: hex(ord(x["char"])),
        "image": lambda x: Image.eval(
            Image.frombytes("F", (128, 127),
                            x["Image Data"], "bit", 4).convert("L"),
            lambda p: p * 16,
        ),
    },
)

B8_Type = RecordType(
    512,
    {
        "Serial Sheet Number": "uint:16",
        "JIS Kanji Code": "hex:16",
        "JIS Typical Reading": "bytes:4",
        "Image Data": "bytes:504",
    },
    {
        "JIS_typical_reading": lambda x: x["JIS Typical Reading"].decode("ascii"),
        "char": lambda x: bytes.fromhex("1b2442" + x["JIS Kanji Code"] + "1b2842").decode("iso2022_jp"),
        "unicode": lambda x: hex(ord(x["char"])),
        "image": lambda x: Image.frombytes("1", (64, 63), x["Image Data"], "raw"),
    },
)

G9_Type = RecordType(
    8199,
    {
        "Serial Sheet Number": "uint:16",
        "JIS Kanji Code": "hex:16",
        "JIS Typical Reading": "bytes:8",
        "Serial Data Number": "uint:32",
        "Evaluation of Individual Character Image": "uint:8",
        "Evaluation of Character Group": "uint:8",
        "Male-Female Code": "uint:8",
        "Age of Writer": "uint:8",
        "Industry Classification Code": "uint:16",
        "Occupation Classification Code": "uint:16",
        "Sheet Gatherring Date": "uint:16",
        "Scanning Date": "uint:16",
        "Sample Position X on Sheet": "uint:8",
        "Sample Position Y on Sheet": "uint:8",
        "(undefined)": "pad:272",
        "Image Data": "bytes:8128",
        "(uncertain)": "pad:56",
    },
    {
        "JIS_kanji_code": lambda x: "1b2442" + x["JIS Kanji Code"] + "1b2842",
        "JIS_typical_reading": lambda x: x["JIS Typical Reading"].decode("ascii"),
        "char": lambda x: bytes.fromhex(x["JIS_kanji_code"]).decode("iso2022_jp"),
        "unicode": lambda x: hex(ord(x["char"])),
        "image": lambda x: Image.eval(
            Image.frombytes("F", (128, 127),
                            x["Image Data"], "bit", 4).convert("L"),
            lambda p: p * 16,
        ),
    },
)

B9_Type = RecordType(
    576,
    {
        "Serial Sheet Number": "uint:16",
        "JIS Kanji Code": "hex:16",
        "JIS Typical Reading": "bytes:4",
        "Image Data": "bytes:504",
        "(uncertain)": "pad:512",
    },
    {
        "JIS_kanji_code": lambda x: "1b2442" + x["JIS Kanji Code"] + "1b2842",
        "JIS_typical_reading": lambda x: x["JIS Typical Reading"].decode("ascii"),
        "char": lambda x: bytes.fromhex(x["JIS_kanji_code"]).decode("iso2022_jp"),
        "unicode": lambda x: hex(ord(x["char"])),
        "image": lambda x: Image.frombytes("1", (64, 63), x["Image Data"], "raw"),
    },
)


def dataset_name_from_path(input_path: Path) -> str:
    # e.g. ETL4/ETL4C -> parent dir name = ETL4
    return input_path.parent.name


if __name__ == "__main__":
    import argparse
    from tqdm.contrib import tenumerate

    parser = argparse.ArgumentParser(
        description="Decompose ETLn files and emit labels.jsonl")
    parser.add_argument("input", help="input file (e.g. ETL4/ETL4C)")
    parser.add_argument("--euc_co59", default="euc_co59.dat",
                        help="path to euc_co59.dat (for ETL2)")
    parser.add_argument("--no_timestamp_prefix", action="store_true",
                        help="do not prefix filenames with timestamp")
    args = parser.parse_args()

    # If user supplies a different euc_co59 path (global is NOT needed at module scope)
    if args.euc_co59 != "euc_co59.dat":
        co59_to_unicode = CO59_to_unicode(args.euc_co59)

    input_path = Path(args.input)

    reader = {
        "ETL1": M_Type,
        "ETL2": K_Type,
        "ETL3": C_Type,
        "ETL4": C_Type,
        "ETL5": C_Type,
        "ETL6": M_Type,
        "ETL7": M_Type,
        "ETL8G": G8_Type,
        "ETL8B": B8_Type,
        "ETL9G": G9_Type,
        "ETL9B": B9_Type,
    }

    dataset = dataset_name_from_path(input_path)
    if dataset not in reader:
        raise SystemExit(f"Unsupported dataset folder name: {dataset}")

    skip_first = dataset in ["ETL8B", "ETL9B"]
    records = reader[dataset].read(input_path, skip_first=skip_first)

    output_dir = input_path.parent / (input_path.name + "_unpack")
    images_dir = output_dir / "images"
    output_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # timestamp like 20260113_000908
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    labels_path = output_dir / "labels.jsonl"

    with open(labels_path, "w", encoding="utf-8") as wf:
        for i, r in tenumerate(records):
            if args.no_timestamp_prefix:
                filename = f"{i:06d}.png"
            else:
                filename = f"{ts}_{i:06d}.png"

            out_img_path = images_dir / filename
            r["image"].save(out_img_path)

            text = r.get("char", "")
            rel_path = f"images/{filename}"
            wf.write(json.dumps(
                {"path": rel_path, "text": text}, ensure_ascii=False) + "\n")

    print(f"[ok] wrote images to: {images_dir}")
    print(f"[ok] wrote labels to: {labels_path}")
