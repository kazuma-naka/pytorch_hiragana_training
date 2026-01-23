#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# build_dataset_from_android_pictures.py  (collect_hiragana_tk.py 寄せ版 + augmentation)

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image, ImageFilter


@dataclass
class BuildConfig:
    src_root: Path  # 例: exported/Pictures/YourAppName
    out_dir: Path  # 例: dataset_hira
    prefix: str = "label_"  # フォルダ名のprefix

    # collect側に合わせる
    target_h: int = 32
    max_w: int = 512
    ink_thresh: int = 250  # collectのtight bboxが 250
    pad: int = 0  # Android側でtrim済みなら 0 推奨
    blur_radius: float = 0.6  # collectのデフォルトに合わせる

    recursive: bool = True
    limit_per_label: int = 0  # 0なら無制限

    # ---- augmentation options ----
    aug_copies: int = 0  # 1枚から何枚増やすか（0なら増やさない）
    rot_deg: float = 0.0  # 回転の最大絶対値（例: 10 → -10..+10）
    scale_min: float = 1.0  # 拡大縮小の最小（例: 0.9）
    scale_max: float = 1.0  # 拡大縮小の最大（例: 1.1）
    translate_px: int = 0  # 平行移動の最大（px, 例: 6 → -6..+6）
    seed: int = 0  # 0なら固定しない。>0なら再現性あり。


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


def _tight_ink_bbox(
    img_L: Image.Image, ink_thresh: int
) -> Optional[Tuple[int, int, int, int]]:
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


def _apply_affine_augment(
    img_L: Image.Image,
    *,
    rot_deg: float,
    scale_min: float,
    scale_max: float,
    translate_px: int,
) -> Image.Image:
    """
    白背景前提で、回転・拡縮・平行移動を 1 回適用する。
    - expand=True で外枠が増えるので、その後に bbox で締め直す前提。
    - fillcolor=255 で白埋め。
    """
    # rotation
    if rot_deg and rot_deg > 0:
        angle = random.uniform(-rot_deg, rot_deg)
    else:
        angle = 0.0

    # scale
    if scale_min != 1.0 or scale_max != 1.0:
        if scale_min <= 0 or scale_max <= 0:
            raise ValueError("scale_min/scale_max must be > 0")
        if scale_min > scale_max:
            raise ValueError("scale_min must be <= scale_max")
        s = random.uniform(scale_min, scale_max)
    else:
        s = 1.0

    # translate
    if translate_px and translate_px > 0:
        dx = random.randint(-translate_px, translate_px)
        dy = random.randint(-translate_px, translate_px)
    else:
        dx, dy = 0, 0

    # 1) rotate with expand (white fill)
    out = img_L.rotate(
        angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=255
    )

    # 2) scale around top-left (simple resize)
    if s != 1.0:
        nw = max(1, int(round(out.size[0] * s)))
        nh = max(1, int(round(out.size[1] * s)))
        out = out.resize((nw, nh), resample=Image.Resampling.BILINEAR)

    # 3) translate: paste onto same-sized canvas (white background)
    if dx != 0 or dy != 0:
        canvas = Image.new("L", out.size, 255)
        canvas.paste(out, (dx, dy))
        out = canvas

    return out


def preprocess_one_collect_like_from_img(
    img_L: Image.Image, cfg: BuildConfig
) -> Image.Image:
    """
    Pathではなく Image(L) を受け取り、collect思想の前処理を実施する。
    """
    # 1) tight bbox
    bbox1 = _tight_ink_bbox(img_L, ink_thresh=int(cfg.ink_thresh))
    if bbox1 is None:
        raise ValueError("Empty image (no ink detected).")
    cropped = _crop_with_pad(img_L, bbox1, pad=int(cfg.pad))

    # 2) crop後に再tight bbox（余白を落とす）
    bbox2 = _tight_ink_bbox(cropped, ink_thresh=int(cfg.ink_thresh))
    if bbox2 is None:
        raise ValueError("Empty after crop (no ink detected).")
    cropped = cropped.crop(bbox2)

    # 3) blur
    if cfg.blur_radius and cfg.blur_radius > 0:
        cropped = cropped.filter(
            ImageFilter.GaussianBlur(radius=float(cfg.blur_radius))
        )

    # 4) 高さ固定へ
    resized = _resize_keep_aspect_to_h(cropped, target_h=int(cfg.target_h))

    # 5) 幅上限
    if resized.size[0] > int(cfg.max_w):
        resized = resized.crop((0, 0, int(cfg.max_w), resized.size[1]))

    return resized


def preprocess_one_collect_like(img_path: Path, cfg: BuildConfig) -> Image.Image:
    """
    既存と同じ：pathから読み込んで前処理する（拡張なしの“元画像用”）。
    """
    img = Image.open(img_path)
    img_L = _ensure_white_bg_then_L(img)
    return preprocess_one_collect_like_from_img(img_L, cfg)


def preprocess_many_with_aug(
    img_path: Path, cfg: BuildConfig
) -> List[Tuple[Image.Image, str]]:
    """
    1枚の入力から:
      - 元画像 1枚
      - aug_copies 枚の拡張画像
    を生成して返す。
    戻り値: [(img_L, variant), ...] variant は "orig" or "aug{i}"
    """
    base = Image.open(img_path)
    base_L = _ensure_white_bg_then_L(base)

    out: List[Tuple[Image.Image, str]] = []
    # orig
    out.append((preprocess_one_collect_like_from_img(base_L, cfg), "orig"))

    if cfg.aug_copies <= 0:
        return out

    # 拡張は「crop済み」状態から行う方が安定するので、
    # まず base_L を一度 tight crop（pad込み）してから augment → bbox締め直し → 以降同じ
    bbox = _tight_ink_bbox(base_L, ink_thresh=int(cfg.ink_thresh))
    if bbox is None:
        return out  # origは上で例外になるので通常ここには来ない
    cropped0 = _crop_with_pad(base_L, bbox, pad=int(cfg.pad))
    bbox0 = _tight_ink_bbox(cropped0, ink_thresh=int(cfg.ink_thresh))
    if bbox0 is None:
        return out
    cropped0 = cropped0.crop(bbox0)

    for i in range(cfg.aug_copies):
        aug = _apply_affine_augment(
            cropped0,
            rot_deg=float(cfg.rot_deg),
            scale_min=float(cfg.scale_min),
            scale_max=float(cfg.scale_max),
            translate_px=int(cfg.translate_px),
        )

        # augment後にbbox締め直し（回転で増えた余白を落とす）
        bb = _tight_ink_bbox(aug, ink_thresh=int(cfg.ink_thresh))
        if bb is None:
            # たまにインクが消える/極端に薄くなるケースはスキップ
            continue
        aug = aug.crop(bb)

        # あとは同じパイプライン（blur→resize→max_w）
        aug_final = preprocess_one_collect_like_from_img(aug, cfg)
        out.append((aug_final, f"aug{i:02d}"))

    return out


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
        text = name[len(cfg.prefix) :]
        if text == "":
            continue
        out.append((text, p))
    out.sort(key=lambda x: x[0])
    return out


def collect_images(folder: Path, recursive: bool) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if recursive:
        paths = [
            p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts
        ]
    else:
        paths = [
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts
        ]
    paths.sort()
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_root",
        type=str,
        required=True,
        help="Androidからコピーした Pictures/<AppName> 相当のフォルダ",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="dataset_hira",
        help="出力datasetフォルダ（labels.jsonl と images/ を作る）",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="label_",
        help="ラベルフォルダのprefix（例: label_あ の label_）",
    )

    # collect寄せのデフォルト
    ap.add_argument("--target_h", type=int, default=32)
    ap.add_argument("--max_w", type=int, default=512)
    ap.add_argument("--ink_thresh", type=int, default=250)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--blur", type=float, default=0.6)

    ap.add_argument("--no_recursive", action="store_true")
    ap.add_argument(
        "--limit_per_label", type=int, default=0, help="各ラベル最大枚数。0なら無制限"
    )

    # ---- augmentation args ----
    ap.add_argument(
        "--aug_copies",
        type=int,
        default=0,
        help="各入力画像から追加生成する枚数（0なら増やさない）",
    )
    ap.add_argument(
        "--rot_deg",
        type=float,
        default=0.0,
        help="回転の最大絶対値（例: 10 → -10..+10）",
    )
    ap.add_argument(
        "--scale_min", type=float, default=1.0, help="スケール最小（例: 0.9）"
    )
    ap.add_argument(
        "--scale_max", type=float, default=1.0, help="スケール最大（例: 1.1）"
    )
    ap.add_argument(
        "--translate_px", type=int, default=0, help="平行移動の最大px（例: 6 → -6..+6）"
    )
    ap.add_argument("--seed", type=int, default=0, help="再現性用seed。0なら固定しない")

    args = ap.parse_args()

    if args.seed and args.seed > 0:
        random.seed(int(args.seed))

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
        aug_copies=int(args.aug_copies),
        rot_deg=float(args.rot_deg),
        scale_min=float(args.scale_min),
        scale_max=float(args.scale_max),
        translate_px=int(args.translate_px),
        seed=int(args.seed),
    )

    out_images = cfg.out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    labels_path = cfg.out_dir / "labels.jsonl"

    label_folders = iter_label_folders(cfg)
    if not label_folders:
        raise RuntimeError(
            f"No label folders found under {cfg.src_root} with prefix '{cfg.prefix}'"
        )

    n_ok = 0
    n_skip = 0
    uid_count = 0

    with labels_path.open("w", encoding="utf-8") as f:
        for text, folder in label_folders:
            paths = collect_images(folder, recursive=cfg.recursive)
            if cfg.limit_per_label > 0:
                paths = paths[: cfg.limit_per_label]

            for p in paths:
                try:
                    # 元 + aug をまとめて生成
                    imgs = preprocess_many_with_aug(p, cfg)

                    for img_L, variant in imgs:
                        uid = f"{_now_id()}_{uid_count:06d}_{variant}"
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
    if cfg.aug_copies > 0:
        print(
            " - augmentation:",
            f"aug_copies={cfg.aug_copies}, rot_deg={cfg.rot_deg}, "
            f"scale=[{cfg.scale_min},{cfg.scale_max}], translate_px={cfg.translate_px}, seed={cfg.seed}",
        )


if __name__ == "__main__":
    main()
