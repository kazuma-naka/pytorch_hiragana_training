#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# unpack.py

from __future__ import annotations

import codecs
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple
import datetime
import json

import bitstring
from PIL import Image, ImageFilter
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


# ---------------- Collect-like preprocess (shared) ----------------

@dataclass
class PreprocessCfg:
    target_h: int = 32
    max_w: int = 512
    ink_thresh: int = 250
    pad: int = 0
    blur_radius: float = 0.6


def _ensure_white_bg_then_L(img: Image.Image) -> Image.Image:
    """
    透過PNGが来ても学習と整合するよう、白背景に合成して L にする。
    （ETL8Bは通常 1/L なのでそのまま L 化される）
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        comp = Image.alpha_composite(bg, rgba)
        return comp.convert("L")
    return img.convert("L")


def _tight_ink_bbox(img_L: Image.Image, ink_thresh: int) -> Optional[Tuple[int, int, int, int]]:
    """
    White bg(255) / black ink(0) を想定し、ink_thresh未満をインクとみなしてbboxを返す。
    """
    mask = img_L.point(lambda p: 255 if p < ink_thresh else 0, mode="L")
    return mask.getbbox()


def _crop_with_pad(img_L: Image.Image, bbox, pad: int) -> Image.Image:
    if bbox is None:
        raise ValueError("Nothing drawn.")
    minx, miny, maxx, maxy = bbox
    minx = max(minx - pad, 0)
    miny = max(miny - pad, 0)
    maxx = min(maxx + pad, img_L.size[0])
    maxy = min(maxy + pad, img_L.size[1])
    return img_L.crop((minx, miny, maxx, maxy))


def _resize_keep_aspect_to_h(img_L: Image.Image, target_h: int) -> Image.Image:
    w, h = img_L.size
    if h <= 0:
        return img_L
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return img_L.resize((new_w, target_h), resample=Image.Resampling.BILINEAR)


def preprocess_collect_like(img: Image.Image, cfg: PreprocessCfg) -> Image.Image:
    """
    build_dataset_from_android_pictures.py と同じ思想:
      1) tight bbox → crop(pad)
      2) もう一度 tight bbox → crop（余白を極小化）
      3) blur（任意）
      4) 高さ target_h にリサイズ（幅は可変）
      5) 幅が max_w 超なら右を切る
    """
    img_L = _ensure_white_bg_then_L(img)

    bbox1 = _tight_ink_bbox(img_L, ink_thresh=int(cfg.ink_thresh))
    if bbox1 is None:
        raise ValueError("Empty image (no ink detected).")
    cropped = _crop_with_pad(img_L, bbox1, pad=int(cfg.pad))

    bbox2 = _tight_ink_bbox(cropped, ink_thresh=int(cfg.ink_thresh))
    if bbox2 is None:
        raise ValueError("Empty after crop (no ink detected).")
    cropped = cropped.crop(bbox2)

    if cfg.blur_radius and cfg.blur_radius > 0:
        cropped = cropped.filter(ImageFilter.GaussianBlur(radius=float(cfg.blur_radius)))

    resized = _resize_keep_aspect_to_h(cropped, target_h=int(cfg.target_h))

    if resized.size[0] > int(cfg.max_w):
        resized = resized.crop((0, 0, int(cfg.max_w), resized.size[1]))

    return resized


# ---------------- ETL8B: label normalization / filtering ----------------

_HIRA_SMALL_TO_BIG = str.maketrans({
    "ぁ": "あ",
    "ぃ": "い",
    "ぅ": "う",
    "ぇ": "え",
    "ぉ": "お",
    "っ": "つ",
    "ゃ": "や",
    "ゅ": "ゆ",
    "ょ": "よ",
    "ゎ": "わ",
    "ゕ": "か",
    "ゖ": "け",
})


def normalize_etl8b_label_text(text: str) -> str:
    """
    ETL8B の labels.jsonl 用:
    - 小書きひらがなを大文字に寄せる（複数文字にも対応）
    """
    if not text:
        return text
    return text.translate(_HIRA_SMALL_TO_BIG)


def _to_hiragana_text(text: str) -> str:
    """
    ひらがな抽出用:
    - 半角→全角
    - カタカナ→ひらがな
    """
    if not text:
        return text
    z = jaconv.han2zen(text)
    return jaconv.kata2hira(z)


def _is_hiragana_or_allowed(text: str) -> bool:
    """
    ETL8B の「ひらがなのみ」フィルタ用。
    1文字想定だが、念のため複数文字にも対応。
    許可:
      - ぁ..ゖ (U+3041..U+3096)
      - ー、。 と空白（必要なら）
    """
    if not text:
        return False

    allowed_extra = {"ー", "、", "。", " "}
    for ch in text:
        cp = ord(ch)
        if ch in allowed_extra:
            continue
        if 0x3041 <= cp <= 0x3096:
            continue
        return False
    return True


# ---------------- Record Types ----------------

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

                for k, v in self.converters.items():
                    try:
                        record[k] = v(record)
                    except Exception as e:
                        print(f"Warning: Error occurred in converting {k}: {e}")
                        continue

                records.append(record)

        return records


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
            Image.frombytes("F", (64, 63), x["Image Data"], "bit", 4).convert("L"),
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
            Image.frombytes("F", (60, 60), x["Image Data"], "bit", 6).convert("L"),
            lambda p: p * 4,
        ),
        "mark_of_style": lambda x: T56(x["Mark of Style"]),
        "contents": lambda x: "".join([T56(b.uint) for b in x["Contents"].cut(6)]),
        "style": lambda x: "".join([T56(b.uint) for b in x["Style"].cut(6)]),
    },
)

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
            Image.frombytes("F", (72, 76), x["Image Data"], "bit", 4).convert("L"),
            lambda p: p * 16,
        ),
        "_char": lambda x: bytes.fromhex(x["JIS Code"]).decode("shift_jis"),
        "char": lambda x: (
            jaconv.kata2hira(jaconv.han2zen(x["_char"])).replace("ぃ", "ゐ").replace("ぇ", "ゑ")
            if x["fourcc"] and x["fourcc"][0] == "H"
            else jaconv.han2zen(x["_char"]).replace("ィ", "ヰ").replace("ェ", "ヱ")
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
            Image.frombytes("F", (128, 127), x["Image Data"], "bit", 4).convert("L"),
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
        # ここは「元のままの生画像」(1bit 64x63)
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
            Image.frombytes("F", (128, 127), x["Image Data"], "bit", 4).convert("L"),
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
    return input_path.parent.name


def resolve_inputs(input_path: Path) -> List[Path]:
    """
    入力がファイルならそれ1つ。
    入力がディレクトリなら配下のファイルを列挙（ETL8INFO は除外）。
    """
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = sorted([p for p in input_path.iterdir() if p.is_file()])
        out = []
        for p in files:
            if p.name.upper() == "ETL8INFO":
                continue
            out.append(p)
        return out

    raise SystemExit(f"Input path not found: {input_path}")


if __name__ == "__main__":
    from tqdm.contrib import tenumerate
    import argparse

    parser = argparse.ArgumentParser(description="Decompose ETLn files and emit labels.jsonl")
    parser.add_argument("input", help="input file or directory (e.g. ETL8B/ETL8B2C1 or ETL8B/)")
    parser.add_argument("--euc_co59", default="euc_co59.dat", help="path to euc_co59.dat (for ETL2)")
    parser.add_argument("--no_timestamp_prefix", action="store_true", help="do not prefix filenames with timestamp")

    # ETL8B options
    parser.add_argument("--etl8b_uppercase", action="store_true",
                        help="ETL8B: normalize small hiragana to big hiragana in labels.jsonl")
    parser.add_argument("--etl8b_collect_like", action="store_true",
                        help="ETL8B: apply collect-like preprocess to output images (tight bbox/blur/resize)")
    parser.add_argument("--etl8b_hiragana_only", action="store_true",
                        help="ETL8B: keep only hiragana labels (katakana is converted to hiragana); skip others")

    # collect-like preprocess params (same defaults as your android builder)
    parser.add_argument("--target_h", type=int, default=32)
    parser.add_argument("--max_w", type=int, default=512)
    parser.add_argument("--ink_thresh", type=int, default=250)
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--blur", type=float, default=0.6)

    args = parser.parse_args()

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

    inputs = resolve_inputs(input_path)
    if not inputs:
        raise SystemExit("No input files found to process.")

    dataset = dataset_name_from_path(inputs[0])
    if dataset not in reader:
        raise SystemExit(f"Unsupported dataset folder name: {dataset}")

    # output dir
    if input_path.is_dir():
        output_dir = input_path / "_unpack"
    else:
        output_dir = input_path.parent / (input_path.name + "_unpack")

    images_dir = output_dir / "images"
    output_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    labels_path = output_dir / "labels.jsonl"

    pp_cfg = PreprocessCfg(
        target_h=int(args.target_h),
        max_w=int(args.max_w),
        ink_thresh=int(args.ink_thresh),
        pad=int(args.pad),
        blur_radius=float(args.blur),
    )

    total_written = 0
    total_seen = 0
    total_filtered = 0

    with open(labels_path, "w", encoding="utf-8") as wf:
        idx = 0
        for in_file in inputs:
            skip_first = dataset in ["ETL8B", "ETL9B"]
            records = reader[dataset].read(in_file, skip_first=skip_first)

            for _, r in tenumerate(records):
                total_seen += 1

                # label text
                text = r.get("char", "") or ""

                # ETL8B: hiragana-only filter
                if dataset == "ETL8B" and args.etl8b_hiragana_only:
                    text = _to_hiragana_text(text)
                    if not _is_hiragana_or_allowed(text):
                        total_filtered += 1
                        continue

                # ETL8B: small->big normalization (optional)
                if dataset == "ETL8B" and args.etl8b_uppercase:
                    text = normalize_etl8b_label_text(text)

                # image (optionally preprocess ETL8B)
                img: Image.Image = r["image"]
                if dataset == "ETL8B" and args.etl8b_collect_like:
                    try:
                        img = preprocess_collect_like(img, pp_cfg)
                    except Exception as e:
                        print(f"[skip] preprocess failed: {in_file} rec={idx} -> {e}")
                        continue

                # filename
                if args.no_timestamp_prefix:
                    filename = f"{idx:06d}.png"
                else:
                    filename = f"{ts}_{idx:06d}.png"

                out_img_path = images_dir / filename
                img.save(out_img_path)

                rel_path = f"images/{filename}"
                wf.write(json.dumps({"path": rel_path, "text": text}, ensure_ascii=False) + "\n")

                idx += 1
                total_written += 1

    print(f"[ok] processed files: {len(inputs)}")
    print(f"[ok] total records seen: {total_seen}")
    if dataset == "ETL8B" and args.etl8b_hiragana_only:
        print(f"[ok] total filtered (non-hiragana): {total_filtered}")
    print(f"[ok] total records written: {total_written}")
    print(f"[ok] wrote images to: {images_dir}")
    print(f"[ok] wrote labels to: {labels_path}")
