#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# build_dataset_from_android_pictures.py  (collect_hiragana_tk.py 寄せ版)

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image, ImageFilter


@dataclass
class BuildConfig:
    src_root: Path               # 例: exported/Pictures/YourAppName
    out_dir: Path                # 例: dataset_hira
    prefix: str = "label_"       # フォルダ名のprefix

    # collect側に合わせる
    target_h: int = 32
    max_w: int = 512
    ink_thresh: int = 250        # collectのtight bboxが 250
    pad: int = 0                 # Android側でtrim済みなら 0 推奨
    blur_radius: float = 0.6     # collectのデフォルトに合わせる

    recursive: bool = True
    limit_per_label: int = 0     # 0なら無制限


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_white_bg_then_L(img: Image.Image) -> Image.Image:
    """
    透過PNGが来ても学習と整合するよう、白背景に合成して L にする。
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


def preprocess_one_collect_like(img_path: Path, cfg: BuildConfig) -> Image.Image:
    """
    collect_hiragana_tk.py の _preprocess_for_save と同じ思想に寄せる:
      1) tight bbox → crop(pad)
      2) もう一度 tight bbox → crop（余白を極小化）
      3) blur（任意）
      4) 高さ target_h にリサイズ（幅は可変）
      5) 幅が max_w 超なら右を切る
    """
    img = Image.open(img_path)
    img_L = _ensure_white_bg_then_L(img)

    # 1) まず全体からtight bbox
    bbox1 = _tight_ink_bbox(img_L, ink_thresh=int(cfg.ink_thresh))
    if bbox1 is None:
        raise ValueError("Empty image (no ink detected).")

    cropped = _crop_with_pad(img_L, bbox1, pad=int(cfg.pad))

    # 2) cropped後に再度tight bbox（collectと同じ：余白を落とす）
    bbox2 = _tight_ink_bbox(cropped, ink_thresh=int(cfg.ink_thresh))
    if bbox2 is None:
        raise ValueError("Empty after crop (no ink detected).")
    cropped = cropped.crop(bbox2)

    # 3) blur（collect寄せ）
    if cfg.blur_radius and cfg.blur_radius > 0:
        cropped = cropped.filter(ImageFilter.GaussianBlur(radius=float(cfg.blur_radius)))

    # 4) 高さ固定へ
    resized = _resize_keep_aspect_to_h(cropped, target_h=int(cfg.target_h))

    # 5) 幅上限
    if resized.size[0] > int(cfg.max_w):
        resized = resized.crop((0, 0, int(cfg.max_w), resized.size[1]))

    # 重要: 右パディングはしない（幅可変で保存）
    return resized


def iter_label_folders(cfg: BuildConfig) -> List[Tuple[str, Path]]:
    """
    src_root配下から label_<text> フォルダを列挙し、(text, folder_path) を返す。
    例: src_root/label_あ
    """
    out: List[Tuple[str, Path]] = []
    if not cfg.src_root.exists():
        raise FileNotFoundError(f"src_root not found: {cfg.src_root}")

    for p in cfg.src_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith(cfg.prefix):
            continue
        text = name[len(cfg.prefix):]
        if text == "":
            continue
        out.append((text, p))
    out.sort(key=lambda x: x[0])
    return out


def collect_images(folder: Path, recursive: bool) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if recursive:
        paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    paths.sort()
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", type=str, required=True,
                    help="Androidからコピーした Pictures/<AppName> 相当のフォルダ")
    ap.add_argument("--out_dir", type=str, default="dataset_hira",
                    help="出力datasetフォルダ（labels.jsonl と images/ を作る）")
    ap.add_argument("--prefix", type=str, default="label_",
                    help="ラベルフォルダのprefix（例: label_あ の label_）")

    # collect寄せのデフォルト
    ap.add_argument("--target_h", type=int, default=32)
    ap.add_argument("--max_w", type=int, default=512)
    ap.add_argument("--ink_thresh", type=int, default=250)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--blur", type=float, default=0.6)

    ap.add_argument("--no_recursive", action="store_true")
    ap.add_argument("--limit_per_label", type=int, default=0,
                    help="各ラベル最大枚数。0なら無制限")

    args = ap.parse_args()

    cfg = BuildConfig(
        src_root=Path(args.src_root),
        out_dir=Path(args.out_dir),
        prefix=str(args.prefix),
        target_h=int(args.target_h),
        max_w=int(args.max_w),
        ink_thresh=int(args.ink_thresh),
        pad=int(args.pad),
        blur_radius=float(args.blur),
        recursive=(not args.no_recursive),
        limit_per_label=int(args.limit_per_label),
    )

    out_images = cfg.out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    labels_path = cfg.out_dir / "labels.jsonl"

    label_folders = iter_label_folders(cfg)
    if not label_folders:
        raise RuntimeError(f"No label folders found under {cfg.src_root} with prefix '{cfg.prefix}'")

    n_ok = 0
    n_skip = 0
    uid_count = 0

    # collectと同様に images/<uid>.png + labels.jsonl（pathはimages配下を参照）
    with labels_path.open("w", encoding="utf-8") as f:
        for text, folder in label_folders:
            paths = collect_images(folder, recursive=cfg.recursive)
            if cfg.limit_per_label > 0:
                paths = paths[:cfg.limit_per_label]

            for p in paths:
                try:
                    img_L = preprocess_one_collect_like(p, cfg)

                    uid = f"{_now_id()}_{uid_count:06d}"
                    rel = Path("images") / f"{uid}.png"
                    out_path = cfg.out_dir / rel
                    img_L.save(out_path)

                    obj = {"path": rel.as_posix(), "text": text}
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

                    n_ok += 1
                    uid_count += 1
                except Exception as e:
                    n_skip += 1
                    print(f"[skip] {p} -> {e}")

    print("Done.")
    print(" - out_dir:", cfg.out_dir)
    print(" - labels:", labels_path)
    print(" - images:", out_images)
    print(f" - ok={n_ok} skip={n_skip}")


if __name__ == "__main__":
    main()
