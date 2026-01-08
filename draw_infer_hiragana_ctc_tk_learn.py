#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# draw_infer_hiragana_ctc_tk_learn.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
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
    return Vocab(itos=obj["itos"])


# ---------------- Preprocess (must match collect/train) ----------------

@dataclass
class InferConfig:
    canvas_size: int = 320
    stroke_width: int = 14
    pad: int = 20
    target_h: int = 32
    max_w: int = 512
    blur_radius: float = 0.6
    # pixels < ink_thresh treated as ink (white bg, black ink)
    ink_thresh: int = 245


def _crop_with_pad(img_L: Image.Image, bbox, pad: int, canvas_size: int) -> Image.Image:
    if bbox is None:
        raise ValueError("Nothing drawn.")
    minx, miny, maxx, maxy = bbox
    minx = max(minx - pad, 0)
    miny = max(miny - pad, 0)
    maxx = min(maxx + pad, canvas_size)
    maxy = min(maxy + pad, canvas_size)
    return img_L.crop((minx, miny, maxx, maxy))


def _resize_keep_aspect_to_h(img_L: Image.Image, target_h: int) -> Image.Image:
    w, h = img_L.size
    if h <= 0:
        return img_L
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return img_L.resize((new_w, target_h), resample=Image.Resampling.BILINEAR)


def _pad_to_max_w(img_L: Image.Image, max_w: int, bg: int = 255) -> Image.Image:
    w, h = img_L.size
    if w >= max_w:
        return img_L.crop((0, 0, max_w, h))
    out = Image.new("L", (max_w, h), bg)
    out.paste(img_L, (0, 0))
    return out


def estimate_valid_width_from_array(arr01: np.ndarray, ink_thresh01: float) -> int:
    """
    arr01: [H,W] float in [0,1], white bg ~1.0, black ink ~0.0
    returns valid width in pixels (>=1)
    """
    ink = arr01 < ink_thresh01
    col = ink.any(axis=0)
    idx = np.where(col)[0]
    if idx.size == 0:
        return int(arr01.shape[1])
    return int(idx[-1]) + 1


def _tight_ink_bbox(img_L: Image.Image, ink_thresh: int = 245):
    """
    White bg(255) / black ink(0) を想定。
    ink_thresh 未満の画素を「インク」とみなし、その外接 bbox を返す。
    """
    mask = img_L.point(lambda p: 255 if p < ink_thresh else 0, mode="L")
    return mask.getbbox()  # None if empty


def preprocess_for_infer(img_L: Image.Image, cfg: InferConfig, bbox) -> Tuple[torch.Tensor, int, Image.Image]:
    """
    collect_hiragana_tk.py 側の「余白ゼロ」前処理と合わせる版。

    returns:
      x: [1,1,H,W] float32 in [0,1]
      valid_w_px: 有効幅(px) = W（パディングしないため）
      out_L: デバッグ用の最終入力画像（L）
    """
    if bbox is None:
        raise ValueError("Nothing drawn.")

    # 1) まず strokes_bbox で粗く切る（余白ゼロ目的なので pad=0）
    cropped = _crop_with_pad(img_L, bbox, pad=0, canvas_size=cfg.canvas_size)

    # 2) ピクセル実体からインク領域 bbox を取り直して “余白ゼロ” に切る
    tb = _tight_ink_bbox(cropped, ink_thresh=int(cfg.ink_thresh))
    if tb is None:
        raise ValueError("Nothing drawn.")
    cropped = cropped.crop(tb)

    # 3) blur（必要なら）※ここで blur すると bbox が広がらない
    if cfg.blur_radius and cfg.blur_radius > 0:
        cropped = cropped.filter(
            ImageFilter.GaussianBlur(radius=float(cfg.blur_radius)))

    # 4) 高さを揃える（幅は可変のまま）
    resized = _resize_keep_aspect_to_h(cropped, cfg.target_h)

    # 5) 念のため max_w を超えるなら右を切る（通常は超えない想定）
    if resized.size[0] > cfg.max_w:
        resized = resized.crop((0, 0, cfg.max_w, resized.size[1]))

    arr = np.array(resized, dtype=np.float32) / 255.0  # [H,W]
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    valid_w_px = int(arr.shape[1])  # パディングしないので「幅=有効幅」
    return x, valid_w_px, resized


def width_to_time_steps(valid_w_px: int) -> int:
    # training used width//4
    return max(1, int(valid_w_px) // 4)


def ctc_greedy_ids(log_probs: torch.Tensor) -> List[int]:
    """
    log_probs: [T,1,V]
    returns argmax ids for B=1, length T
    """
    ids = log_probs.argmax(dim=-1)  # [T,1]
    return ids[:, 0].tolist()


# ---------------- App ----------------

class DrawInferApp:
    def __init__(self, model_ts: Path, vocab_json: Path, cfg: InferConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device

        self.model = torch.jit.load(str(model_ts), map_location=device).eval()
        self.vocab = load_vocab(vocab_json)

        self.root = tk.Tk()
        self.root.title("Draw -> Hiragana CTC Inference (TorchScript)")

        self.canvas = tk.Canvas(
            self.root,
            width=cfg.canvas_size,
            height=cfg.canvas_size,
            bg="white",
            cursor="cross",
        )
        self.canvas.grid(row=0, column=0, columnspan=6, padx=10, pady=10)

        self.img = Image.new("L", (cfg.canvas_size, cfg.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.img)

        self.last: Optional[Tuple[int, int]] = None
        self.strokes_bbox = None

        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)

        tk.Button(self.root, text="Infer", command=self.infer).grid(
            row=1, column=0, sticky="ew", padx=10, pady=5)
        tk.Button(self.root, text="Clear", command=self.clear).grid(
            row=1, column=1, sticky="ew", padx=10, pady=5)
        tk.Button(self.root, text="Save debug", command=self.save_debug).grid(
            row=1, column=2, sticky="ew", padx=10, pady=5)
        tk.Button(self.root, text="Quit", command=self.root.quit).grid(
            row=1, column=3, sticky="ew", padx=10, pady=5)

        self.result_var = tk.StringVar(
            value="Draw hiragana, then click Infer.")
        tk.Label(self.root, textvariable=self.result_var).grid(
            row=2, column=0, columnspan=6, padx=10, pady=10)

        self._last_debug_padded: Optional[Image.Image] = None

    def update_bbox(self, x: int, y: int):
        r = self.cfg.stroke_width // 2 + 2
        box = (x - r, y - r, x + r, y + r)
        if self.strokes_bbox is None:
            self.strokes_bbox = box
        else:
            minx = min(self.strokes_bbox[0], box[0])
            miny = min(self.strokes_bbox[1], box[1])
            maxx = max(self.strokes_bbox[2], box[2])
            maxy = max(self.strokes_bbox[3], box[3])
            self.strokes_bbox = (minx, miny, maxx, maxy)

    def on_down(self, event):
        self.last = (event.x, event.y)
        self.update_bbox(event.x, event.y)

    def on_move(self, event):
        if self.last is None:
            return
        x0, y0 = self.last
        x1, y1 = event.x, event.y

        self.canvas.create_line(
            x0, y0, x1, y1,
            fill="black",
            width=self.cfg.stroke_width,
            capstyle=tk.ROUND,
            smooth=True,
            splinesteps=36,
        )
        self.draw.line((x0, y0, x1, y1), fill=0, width=self.cfg.stroke_width)

        self.last = (x1, y1)
        self.update_bbox(x1, y1)

    def on_up(self, event):
        self.last = None

    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new(
            "L", (self.cfg.canvas_size, self.cfg.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.img)
        self.strokes_bbox = None
        self._last_debug_padded = None
        self.result_var.set("Cleared.")

    def infer(self):
        try:
            x, valid_w_px, padded = preprocess_for_infer(
                self.img, self.cfg, self.strokes_bbox)
            self._last_debug_padded = padded

            x = x.to(self.device)

            with torch.no_grad():
                log_probs = self.model(x)  # [T,1,V]

            # cut to valid time steps
            T_valid = width_to_time_steps(valid_w_px)
            T_valid = min(T_valid, int(log_probs.shape[0]))
            log_probs = log_probs[:T_valid]

            ids = ctc_greedy_ids(log_probs)
            text = self.vocab.decode_greedy_ctc(ids)
            self.result_var.set(f"Text: {text}")
        except Exception as e:
            self.result_var.set(f"Error: {e}")

    def save_debug(self):
        try:
            if self._last_debug_padded is None:
                _, _, padded = preprocess_for_infer(
                    self.img, self.cfg, self.strokes_bbox)
                self._last_debug_padded = padded
            out = Path("debug_hira_ctc_input.png")
            self._last_debug_padded.save(out)
            self.result_var.set(f"Saved {out}")
        except Exception as e:
            self.result_var.set(f"Error: {e}")

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="Path to TorchScript model (.pt)")
    ap.add_argument("--vocab", type=str, required=True,
                    help="Path to vocab.json")
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda"])

    ap.add_argument("--canvas_size", type=int, default=320)
    ap.add_argument("--stroke_width", type=int, default=14)
    ap.add_argument("--pad", type=int, default=20)
    ap.add_argument("--target_h", type=int, default=32)
    ap.add_argument("--max_w", type=int, default=512)
    ap.add_argument("--blur", type=float, default=0.6)
    ap.add_argument("--ink_thresh", type=int, default=245)

    args = ap.parse_args()

    device = "cuda" if (
        args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    cfg = InferConfig(
        canvas_size=int(args.canvas_size),
        stroke_width=int(args.stroke_width),
        pad=int(args.pad),
        target_h=int(args.target_h),
        max_w=int(args.max_w),
        blur_radius=float(args.blur),
        ink_thresh=int(args.ink_thresh),
    )

    app = DrawInferApp(Path(args.model), Path(args.vocab), cfg, device=device)
    app.run()


if __name__ == "__main__":
    main()
