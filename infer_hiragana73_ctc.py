#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# infer_hiragana73_ctc.py

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch


# ---------------- Vocab ----------------

@dataclass
class Vocab:
    itos: List[str]  # index -> char (1..), blank=0

    @property
    def blank(self) -> int:
        return 0

    def decode_greedy_ctc(self, ids: List[int]) -> str:
        out: List[str] = []
        prev = None
        for i in ids:
            if i == self.blank:
                prev = i
                continue
            if prev == i:
                continue
            out.append(self.itos[i - 1])
            prev = i
        return "".join(out)


def load_vocab(path: Path) -> Vocab:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if "itos" not in obj or not isinstance(obj["itos"], list):
        raise ValueError(f"Invalid vocab.json format: {path}")
    return Vocab(itos=[str(x) for x in obj["itos"]])


# ---------------- Preprocess (match training) ----------------

def _tight_bbox_of_ink(img_L: Image.Image, ink_thresh: int) -> Optional[Tuple[int, int, int, int]]:
    """
    White bg(255) / black ink(0) を想定。
    ink_thresh 未満の画素を「インク」とみなし、その外接 bbox を返す。
    """
    a = np.array(img_L, dtype=np.uint8)
    ink = a < int(ink_thresh)
    ys, xs = np.where(ink)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1)


def preprocess_to_fixed_canvas(
    img_path: Path,
    target_h: int,
    max_w: int,
    pad: int,
    ink_thresh: int,
    invert_auto: bool = True,
) -> Tuple[torch.Tensor, int, Image.Image]:
    """
    学習側（hiragana73 train）と同じ思想：
      - 必要なら自動反転
      - インク bbox をタイトクロップ(+pad)
      - 高さ target_h へリサイズ（アスペクト維持）
      - 左詰めで max_w に白パディング
    Returns:
      x: [1,1,target_h,max_w] float32 in [0,1]
      valid_w_px: 元の（リサイズ後）有効幅(px)
      out_L: デバッグ用（max_w,target_h）の最終入力画像
    """
    img = Image.open(img_path).convert("L")

    if invert_auto:
        m = float(np.array(img, dtype=np.uint8).mean())
        # 背景が暗い（黒地に白線など）と推定したら反転して白地黒インクへ寄せる
        if m < 127.0:
            img = Image.fromarray(
                255 - np.array(img, dtype=np.uint8), mode="L")

    bbox = _tight_bbox_of_ink(img, ink_thresh=ink_thresh)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(img.size[0], x1 + pad)
        y1 = min(img.size[1], y1 + pad)
        img = img.crop((x0, y0, x1, y1))

    # resize to target_h
    w, h = img.size
    h = max(1, h)
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    if new_w > max_w:
        new_w = max_w
    img = img.resize((new_w, target_h), resample=Image.BILINEAR)

    # pad to max_w (white)
    canvas = Image.new("L", (max_w, target_h), color=255)
    canvas.paste(img, (0, 0))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    valid_w_px = int(new_w)
    return x, valid_w_px, canvas


def width_to_time_steps(valid_w_px: int) -> int:
    # train で valid_width//4 を使っていた想定
    return max(1, int(valid_w_px) // 4)


def ctc_greedy_ids(log_probs: torch.Tensor) -> List[int]:
    """
    log_probs: [T,1,V] (TorchScript model output)
    returns argmax ids for B=1
    """
    ids = log_probs.argmax(dim=-1)  # [T,1]
    return ids[:, 0].tolist()


def mean_prob_over_time(log_probs: torch.Tensor) -> torch.Tensor:
    """
    log_probs: [T,1,V]
    return: [V] (time-mean probs)
    """
    probs = torch.exp(log_probs[:, 0, :])  # [T,V]
    return probs.mean(dim=0)               # [V]


# ---------------- IO helpers ----------------

def list_images_in_dir(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    out.sort()
    return out


# ---------------- Inference ----------------

@dataclass
class InferConfig:
    target_h: int = 32
    max_w: int = 256
    pad: int = 6
    ink_thresh: int = 245
    invert_auto: bool = True
    topk: int = 5


@torch.no_grad()
def infer_one(
    model: torch.jit.ScriptModule,
    vocab: Vocab,
    img_path: Path,
    cfg: InferConfig,
    device: str,
) -> dict:
    x, valid_w_px, debug_img = preprocess_to_fixed_canvas(
        img_path=img_path,
        target_h=cfg.target_h,
        max_w=cfg.max_w,
        pad=cfg.pad,
        ink_thresh=cfg.ink_thresh,
        invert_auto=cfg.invert_auto,
    )

    x = x.to(device)
    log_probs = model(x)  # [T,1,V]

    # cut to valid time steps
    T_valid = width_to_time_steps(valid_w_px)
    T_valid = min(T_valid, int(log_probs.shape[0]))
    log_probs = log_probs[:T_valid]

    ids = ctc_greedy_ids(log_probs)
    text = vocab.decode_greedy_ctc(ids)

    # confidence (time-mean prob max)
    mp = mean_prob_over_time(log_probs)  # [V]
    conf = float(mp.max().item())

    # top-k tokens (including blank)
    k = max(1, int(cfg.topk))
    k = min(k, int(mp.shape[0]))
    topv, topi = torch.topk(mp, k=k)

    def id_to_token(i: int) -> str:
        if i == 0:
            return "[BLANK]"
        j = i - 1
        if 0 <= j < len(vocab.itos):
            return vocab.itos[j]
        return f"[ID{i}]"

    topk_list = []
    for v, i in zip(topv.tolist(), topi.tolist()):
        topk_list.append({"token": id_to_token(int(i)), "prob": float(v)})

    return {
        "path": str(img_path),
        "pred": text,
        "conf": conf,
        "valid_w_px": int(valid_w_px),
        "T_valid": int(T_valid),
        "topk": topk_list,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="TorchScript model (.pt)")
    ap.add_argument("--vocab", type=str, required=True,
                    help="vocab.json (itos list)")
    ap.add_argument("--image", type=str, default="", help="Single image path")
    ap.add_argument("--dir", type=str, default="",
                    help="Directory to infer recursively")
    ap.add_argument("--out_jsonl", type=str, default="",
                    help="Write results as JSONL")
    ap.add_argument("--save_debug_dir", type=str, default="",
                    help="Save preprocessed input images here (optional)")

    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda"])

    # preprocess (match training defaults)
    ap.add_argument("--target_h", type=int, default=32)
    ap.add_argument("--max_w", type=int, default=256)
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--ink_thresh", type=int, default=245)
    ap.add_argument("--no_invert_auto", action="store_true")

    # output
    ap.add_argument("--topk", type=int, default=5)

    args = ap.parse_args()

    if not args.image and not args.dir:
        raise SystemExit("ERROR: specify --image or --dir")

    device = "cuda" if (
        args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    model = torch.jit.load(str(Path(args.model)), map_location=device).eval()
    vocab = load_vocab(Path(args.vocab))

    cfg = InferConfig(
        target_h=int(args.target_h),
        max_w=int(args.max_w),
        pad=int(args.pad),
        ink_thresh=int(args.ink_thresh),
        invert_auto=(not args.no_invert_auto),
        topk=int(args.topk),
    )

    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else None
    save_debug_dir = Path(args.save_debug_dir) if args.save_debug_dir else None
    if save_debug_dir:
        save_debug_dir.mkdir(parents=True, exist_ok=True)
    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    results: List[dict] = []

    if args.image:
        img_path = Path(args.image)
        r = infer_one(model, vocab, img_path, cfg, device=device)
        results.append(r)

        print(f"path: {r['path']}")
        print(
            f"pred: {r['pred']}  conf: {r['conf']:.4f}  (valid_w={r['valid_w_px']}px, T={r['T_valid']})")
        print("topk:")
        for t in r["topk"]:
            print(f"  - {t['token']}: {t['prob']:.4f}")

        if save_debug_dir:
            # save preprocessed canvas for inspection
            _, _, dbg = preprocess_to_fixed_canvas(
                img_path=img_path,
                target_h=cfg.target_h,
                max_w=cfg.max_w,
                pad=cfg.pad,
                ink_thresh=cfg.ink_thresh,
                invert_auto=cfg.invert_auto,
            )
            outp = save_debug_dir / (img_path.stem + "_pre.png")
            dbg.save(outp)

    if args.dir:
        root = Path(args.dir)
        files = list_images_in_dir(root)
        if not files:
            raise SystemExit(f"ERROR: no images found under {root}")

        for p in files:
            r = infer_one(model, vocab, p, cfg, device=device)
            results.append(r)
            print(f"{r['pred']}\t{r['conf']:.4f}\t{p}")

            if save_debug_dir:
                _, _, dbg = preprocess_to_fixed_canvas(
                    img_path=p,
                    target_h=cfg.target_h,
                    max_w=cfg.max_w,
                    pad=cfg.pad,
                    ink_thresh=cfg.ink_thresh,
                    invert_auto=cfg.invert_auto,
                )
                outp = save_debug_dir / (p.stem + "_pre.png")
                dbg.save(outp)

    if out_jsonl:
        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"saved: {out_jsonl}")


if __name__ == "__main__":
    main()
