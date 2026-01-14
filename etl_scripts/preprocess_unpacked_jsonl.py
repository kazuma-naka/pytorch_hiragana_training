#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# preprocess_unpacked_jsonl.py
#
# Purpose:
#   Normalize "unpack.py generated dataset" into the same training format as
#   collect_hiragana_tk.py output:
#     - white background + black ink
#     - tight ink bbox crop (pad=0)
#     - optional blur
#     - resize keeping aspect to target_h
#     - clamp width to max_w
#
# Input:
#   An "unpack directory" that contains:
#     - labels.jsonl  ({"path":"images/xxx.png","text":"..."} )
#     - images/...
#
# Output:
#   out_dir/
#     images/...(same relative paths as input by default)
#     labels.jsonl   (same schema)
#     meta.json      (params + stats)
#
# Notes:
#   ETL images sometimes have "dark background + light ink" (need invert).
#   Also, a faint border / frame may remain if the background is gray-ish.
#   This script includes "white background normalization" (Otsu-based) to
#   force background to 255 before bbox cropping, which removes the frame.
#
# Extra (requested):
#   Exclude samples whose label is one of: ゐ, ゑ, ヰ, ヱ  (option: --exclude_iwiwe)

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps


# ---------------- Config ----------------

@dataclass
class PrepConfig:
    target_h: int = 32
    max_w: int = 512
    blur_radius: float = 0.6

    # bbox uses: pixel < ink_thresh == ink (after normalization)
    ink_thresh: int = 250

    # invert: auto|yes|no
    invert: str = "auto"

    # percent cutoff; 0 disables
    autocontrast_cutoff: float = 1.0

    # force grayscale
    force_mode_L: bool = True

    # keep same rel paths under out_dir
    preserve_paths: bool = True

    # white background normalization: auto|yes|no
    # - yes: always whiten background with Otsu method
    # - no: disable
    # - auto: enable only when background isn't already "mostly white"
    white_bg: str = "auto"

    # auto mode thresholds
    # ratio of pixels >= 250 treated "already white"
    white_bg_auto_min_white_ratio: float = 0.95

    # exclude labels: ゐ ゑ ヰ ヱ
    exclude_iwiwe: bool = True


# ---------------- Utilities ----------------

def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_jsonl(p: Path) -> Iterable[Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def write_jsonl(p: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _safe_rel_image_path(rel: str) -> Path:
    rel = (rel or "").replace("\\", "/").lstrip("/")
    p = Path(rel)
    if ".." in p.parts:
        raise ValueError(f"Invalid path (contains ..): {rel}")
    return p


def _tight_ink_bbox(img_L: Image.Image, ink_thresh: int) -> Optional[Tuple[int, int, int, int]]:
    # mask: ink=255, bg=0
    mask = img_L.point(lambda p: 255 if p < ink_thresh else 0, mode="L")
    return mask.getbbox()


def _resize_keep_aspect_to_h(img_L: Image.Image, target_h: int) -> Image.Image:
    w, h = img_L.size
    if h <= 0 or target_h <= 0:
        return img_L
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return img_L.resize((new_w, target_h), resample=Image.Resampling.BILINEAR)


def _maybe_invert_auto(im_L: Image.Image) -> Image.Image:
    """
    Auto decide inversion based on image statistics.
    - If background is dark (median < 128), invert it.
    """
    arr = np.array(im_L, dtype=np.uint8)
    med = float(np.median(arr))
    if med < 128.0:
        return ImageOps.invert(im_L)
    return im_L


# ---------------- White background normalization (frame removal) ----------------

def _otsu_threshold_u8(arr_u8: np.ndarray) -> int:
    """
    Otsu threshold for uint8 grayscale image.
    Returns threshold t (0..255).
    """
    hist = np.bincount(arr_u8.reshape(-1), minlength=256)
    total = int(arr_u8.size)
    if total <= 0:
        return 127

    sum_total = int(np.dot(np.arange(256), hist))

    sumB = 0
    wB = 0
    varMax = -1.0
    thr = 0

    for t in range(256):
        wB += int(hist[t])
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * int(hist[t])

        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        varBetween = wB * wF * (mB - mF) ** 2

        if varBetween > varMax:
            varMax = varBetween
            thr = t

    return int(thr)


def _should_whiten_background_auto(arr_u8: np.ndarray, min_white_ratio: float) -> bool:
    """
    Decide whether to apply background whitening in 'auto' mode.
    If the image is already mostly white, skip whitening.
    """
    if arr_u8.size == 0:
        return False
    white_ratio = float(np.mean(arr_u8 >= 250))
    return white_ratio < float(min_white_ratio)


def _whiten_background_otsu(im_L: Image.Image) -> Tuple[Image.Image, int]:
    """
    Force background to 255 based on Otsu threshold.
    - Compute Otsu threshold thr.
    - Foreground(ink) = pixels < thr
    - Background = 255 fixed
    - Foreground is linearly mapped [min_ink..thr] -> [0..255] to boost contrast
    Returns: (image, thr)
    """
    arr = np.asarray(im_L, dtype=np.uint8)
    if arr.size == 0:
        return (Image.new("L", (1, 1), 255), 127)

    thr = _otsu_threshold_u8(arr)
    fg = arr < thr

    if not np.any(fg):
        # no ink; return blank
        return (Image.new("L", (1, max(1, im_L.size[1])), 255), thr)

    min_ink = int(arr[fg].min())
    denom = max(1, thr - min_ink)

    out = arr.astype(np.float32)
    out = (out - float(min_ink)) * (255.0 / float(denom))
    out = np.clip(out, 0, 255)
    out[~fg] = 255.0

    return (Image.fromarray(out.astype(np.uint8)), thr)


def normalize_like_collector(img: Image.Image, cfg: PrepConfig) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Convert grayscale samples into the same "collector" style.
    Returns: (normalized_image, debug_info)
    """
    debug: Dict[str, Any] = {}

    im = img
    if cfg.force_mode_L and im.mode != "L":
        im = im.convert("L")

    inv = (cfg.invert or "auto").strip().lower()
    if inv not in ("auto", "yes", "no"):
        inv = "auto"
    debug["invert_mode"] = inv

    if inv == "yes":
        im = ImageOps.invert(im)
        debug["inverted"] = True
    elif inv == "auto":
        before_med = float(np.median(np.asarray(im, dtype=np.uint8)))
        im2 = _maybe_invert_auto(im)
        after_med = float(np.median(np.asarray(im2, dtype=np.uint8)))
        im = im2
        debug["auto_invert_before_median"] = before_med
        debug["auto_invert_after_median"] = after_med
        debug["inverted"] = (after_med != before_med)
    else:
        debug["inverted"] = False

    if cfg.autocontrast_cutoff and cfg.autocontrast_cutoff > 0:
        im = ImageOps.autocontrast(im, cutoff=float(cfg.autocontrast_cutoff))
        debug["autocontrast_cutoff"] = float(cfg.autocontrast_cutoff)
    else:
        debug["autocontrast_cutoff"] = 0.0

    # ---- white background normalization (frame removal) ----
    wb = (cfg.white_bg or "auto").strip().lower()
    if wb not in ("auto", "yes", "no"):
        wb = "auto"
    debug["white_bg_mode"] = wb

    applied_wb = False
    otsu_thr = None

    if wb == "yes":
        im, otsu_thr = _whiten_background_otsu(im)
        applied_wb = True
    elif wb == "auto":
        arr = np.asarray(im, dtype=np.uint8)
        if _should_whiten_background_auto(arr, cfg.white_bg_auto_min_white_ratio):
            im, otsu_thr = _whiten_background_otsu(im)
            applied_wb = True

    debug["white_bg_applied"] = applied_wb
    if otsu_thr is not None:
        debug["otsu_thr"] = int(otsu_thr)

    tb = _tight_ink_bbox(im, ink_thresh=int(cfg.ink_thresh))
    debug["bbox1"] = tb
    if tb is None:
        return (Image.new("L", (1, int(cfg.target_h)), 255), debug)

    cropped = im.crop(tb)

    # re-crop
    tb2 = _tight_ink_bbox(cropped, ink_thresh=int(cfg.ink_thresh))
    debug["bbox2"] = tb2
    if tb2 is not None:
        cropped = cropped.crop(tb2)

    if cfg.blur_radius and cfg.blur_radius > 0:
        cropped = cropped.filter(
            ImageFilter.GaussianBlur(radius=float(cfg.blur_radius)))
        debug["blur_radius"] = float(cfg.blur_radius)
    else:
        debug["blur_radius"] = 0.0

    resized = _resize_keep_aspect_to_h(cropped, int(cfg.target_h))
    debug["resized_size"] = list(resized.size)

    if resized.size[0] > int(cfg.max_w):
        resized = resized.crop((0, 0, int(cfg.max_w), resized.size[1]))
        debug["clamped_to_max_w"] = True
    else:
        debug["clamped_to_max_w"] = False

    return (resized, debug)


def _find_labels_jsonl(in_dir: Path, labels_arg: str) -> Path:
    if labels_arg and labels_arg.strip():
        p = Path(labels_arg)
        if p.is_dir():
            p = p / "labels.jsonl"
        if not p.exists():
            raise SystemExit(f"labels not found: {p}")
        return p

    cand = in_dir / "labels.jsonl"
    if cand.exists():
        return cand

    cands = list(in_dir.glob("*/labels.jsonl"))
    if cands:
        return cands[0]

    raise SystemExit(f"labels.jsonl not found under: {in_dir}")


def process_one(
    in_root: Path,
    out_root: Path,
    rec: Dict[str, Any],
    cfg: PrepConfig,
    seq_index: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    rel = str(rec.get("path", "")).strip()
    text = str(rec.get("text", "")).strip()
    if not rel or not text:
        return None, None

    # ★ requested exclusion (option 2)
    if cfg.exclude_iwiwe and text in ("ゐ", "ゑ", "ヰ", "ヱ"):
        return None, None

    try:
        rel_p = _safe_rel_image_path(rel)
    except Exception:
        return None, None

    src = in_root / rel_p
    if not src.exists():
        return None, None

    try:
        im = Image.open(src)
        im.load()
    except Exception:
        return None, None

    out_img, dbg = normalize_like_collector(im, cfg)

    if cfg.preserve_paths:
        dst_rel = rel_p
    else:
        dst_rel = Path("images") / f"{seq_index:06d}.png"

    dst = out_root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        out_img.save(dst)
    except Exception:
        return None, None

    out_rec = {"path": str(dst_rel).replace("\\", "/"), "text": text}
    dbg_out = {
        "src": str(rel_p).replace("\\", "/"),
        "dst": str(dst_rel).replace("\\", "/"),
        "text": text,
        **(dbg or {}),
    }
    return out_rec, dbg_out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Normalize unpack.py output (labels.jsonl + images/) into collect_hiragana_tk.py-like dataset."
    )

    ap.add_argument("--in_dir", type=str, required=True,
                    help="unpack output directory (e.g. ETL4/ETL4C_unpack)")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="output dataset directory (e.g. dataset_etl4c_norm)")

    ap.add_argument("--labels", type=str, default="",
                    help="optional path to input labels.jsonl (default: auto-detect in in_dir)")

    ap.add_argument("--target_h", type=int, default=32)
    ap.add_argument("--max_w", type=int, default=512)
    ap.add_argument("--blur", type=float, default=0.6)
    ap.add_argument("--ink_thresh", type=int, default=250)

    ap.add_argument("--invert", type=str, default="auto", help="auto|yes|no")
    ap.add_argument("--autocontrast", type=float, default=1.0,
                    help="percent cutoff; 0 disables")

    ap.add_argument("--white_bg", type=str, default="auto",
                    help="white background normalization: auto|yes|no (removes faint frames)")
    ap.add_argument("--white_bg_auto_min_white_ratio", type=float, default=0.95,
                    help="(auto mode) if ratio of pixels >=250 is below this, apply whitening")

    ap.add_argument("--rewrite_paths", action="store_true",
                    help="rewrite output filenames to images/000000.png... instead of preserving input paths")

    ap.add_argument("--log_every", type=int, default=2000)

    ap.add_argument("--debug_jsonl", type=str, default="",
                    help="optional path to write per-sample debug jsonl (e.g. out_dir/debug.jsonl)")

    # ★ requested exclusion flag (default ON)
    ap.add_argument("--exclude_iwiwe", action="store_true",
                    help="exclude labels: ゐ, ゑ, ヰ, ヱ (default: ON)")
    ap.add_argument("--include_iwiwe", action="store_true",
                    help="do NOT exclude ゐ/ゑ/ヰ/ヱ (overrides --exclude_iwiwe)")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        raise SystemExit(f"in_dir not found: {in_dir}")

    labels_p = _find_labels_jsonl(in_dir, args.labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    # default: exclude ON
    exclude_iwiwe = True
    if args.exclude_iwiwe:
        exclude_iwiwe = True
    if args.include_iwiwe:
        exclude_iwiwe = False

    cfg = PrepConfig(
        target_h=int(args.target_h),
        max_w=int(args.max_w),
        blur_radius=float(args.blur),
        ink_thresh=int(args.ink_thresh),
        invert=str(args.invert),
        autocontrast_cutoff=float(args.autocontrast),
        force_mode_L=True,
        preserve_paths=(not bool(args.rewrite_paths)),
        white_bg=str(args.white_bg),
        white_bg_auto_min_white_ratio=float(
            args.white_bg_auto_min_white_ratio),
        exclude_iwiwe=bool(exclude_iwiwe),
    )

    rows_in = list(read_jsonl(labels_p))
    total = len(rows_in)

    out_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    missing = 0
    failed = 0
    skipped_iwiwe = 0

    white_bg_applied_count = 0

    t0 = time.time()
    for i, rec in enumerate(rows_in, 1):
        # detect skip reason for counters
        txt = str(rec.get("text", "")).strip()
        if cfg.exclude_iwiwe and txt in ("ゐ", "ゑ", "ヰ", "ヱ"):
            skipped_iwiwe += 1
            continue

        out_rec, dbg = process_one(
            in_dir, out_dir, rec, cfg, seq_index=len(out_rows))
        if out_rec is None:
            rel = str(rec.get("path", "")).strip()
            if rel:
                try:
                    rel_p = _safe_rel_image_path(rel)
                    if not (in_dir / rel_p).exists():
                        missing += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
            else:
                failed += 1
            continue

        out_rows.append(out_rec)

        if dbg is not None:
            debug_rows.append(dbg)
            if bool(dbg.get("white_bg_applied", False)):
                white_bg_applied_count += 1

        if args.log_every > 0 and (i % int(args.log_every) == 0):
            dt = time.time() - t0
            print(
                f"[{i}/{total}] ok={len(out_rows)} missing={missing} failed={failed} "
                f"skipped_iwiwe={skipped_iwiwe} white_bg_applied={white_bg_applied_count} elapsed={dt:.1f}s",
                file=sys.stderr
            )

    out_labels = out_dir / "labels.jsonl"
    write_jsonl(out_labels, out_rows)

    if args.debug_jsonl and args.debug_jsonl.strip():
        dbg_p = Path(args.debug_jsonl)
        if dbg_p.is_dir():
            dbg_p = dbg_p / "debug.jsonl"
        dbg_p.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(dbg_p, debug_rows)

    meta = {
        "created_at": _now(),
        "input": {
            "in_dir": str(in_dir),
            "labels": str(labels_p),
            "total_records": total,
        },
        "output": {
            "out_dir": str(out_dir),
            "labels": str(out_labels),
            "saved_records": len(out_rows),
            "missing_images": missing,
            "failed_records": failed,
            "preserve_paths": cfg.preserve_paths,
        },
        "preprocess": {
            "target_h": cfg.target_h,
            "max_w": cfg.max_w,
            "blur_radius": cfg.blur_radius,
            "ink_thresh": cfg.ink_thresh,
            "invert": cfg.invert,
            "autocontrast_cutoff": cfg.autocontrast_cutoff,
            "white_bg": cfg.white_bg,
            "white_bg_auto_min_white_ratio": cfg.white_bg_auto_min_white_ratio,
            "white_bg_applied_count": white_bg_applied_count,
            "exclude_iwiwe": cfg.exclude_iwiwe,
            "skipped_iwiwe": skipped_iwiwe,
        },
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    dt = time.time() - t0
    print(
        f"Done. saved={len(out_rows)} / total={total} missing={missing} failed={failed} "
        f"skipped_iwiwe={skipped_iwiwe} white_bg_applied={white_bg_applied_count} elapsed={dt:.1f}s"
    )
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
