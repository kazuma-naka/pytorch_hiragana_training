#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# draw_infer_hiragana73_ctc_tk.py

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
    return Vocab(itos=[str(x) for x in obj["itos"]])


# ---------------- Preprocess (match hiragana73 training) ----------------

@dataclass
class InferConfig:
    canvas_size: int = 320
    stroke_width: int = 14

    # training side params (must match)
    target_h: int = 32
    max_w: int = 256
    pad: int = 6
    ink_thresh: int = 245
    invert_auto: bool = True

    # optional blur for drawn strokes
    blur_radius: float = 0.6


def _crop_with_pad(img_L: Image.Image, bbox, pad: int, canvas_size: int) -> Image.Image:
    if bbox is None:
        raise ValueError("Nothing drawn.")
    minx, miny, maxx, maxy = bbox
    minx = max(minx - pad, 0)
    miny = max(miny - pad, 0)
    maxx = min(maxx + pad, canvas_size)
    maxy = min(maxy + pad, canvas_size)
    return img_L.crop((minx, miny, maxx, maxy))


def _tight_ink_bbox(img_L: Image.Image, ink_thresh: int):
    """
    White bg(255) / black ink(0) を想定。
    ink_thresh 未満の画素を「インク」とみなし、その外接 bbox を返す。
    """
    mask = img_L.point(lambda p: 255 if p < ink_thresh else 0, mode="L")
    return mask.getbbox()  # None if empty


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


def preprocess_for_infer_hiragana73(
    img_L: Image.Image,
    cfg: InferConfig,
    strokes_bbox,
) -> Tuple[torch.Tensor, int, Image.Image]:
    """
    hiragana73 train スクリプトの preprocess_to_fixed_canvas に寄せる：
      - 必要なら自動反転（手書きは通常不要だがオプション保持）
      - インク bbox タイトクロップ(+pad)
      - target_h へリサイズ
      - max_w へ白パディング（ここが重要）
    Returns:
      x: [1,1,target_h,max_w]
      valid_w_px: リサイズ後の有効幅(px)（time-step切り詰めに使う）
      out_L: デバッグ用（最終入力）
    """
    if strokes_bbox is None:
        raise ValueError("Nothing drawn.")

    # 0) strokes_bboxで粗く切る（キャンバス外余白削り）
    cropped = _crop_with_pad(img_L, strokes_bbox, pad=0,
                             canvas_size=cfg.canvas_size)

    # 1) インク領域 bbox で “余白ゼロ” に切る
    tb = _tight_ink_bbox(cropped, ink_thresh=int(cfg.ink_thresh))
    if tb is None:
        raise ValueError("Nothing drawn.")
    cropped = cropped.crop(tb)

    # 2) blur（必要なら）
    if cfg.blur_radius and cfg.blur_radius > 0:
        cropped = cropped.filter(
            ImageFilter.GaussianBlur(radius=float(cfg.blur_radius)))

    # 3) 自動反転（念のため）
    if cfg.invert_auto:
        m = float(np.array(cropped, dtype=np.uint8).mean())
        if m < 127.0:
            cropped = Image.fromarray(
                255 - np.array(cropped, dtype=np.uint8), mode="L")

    # 4) target_h に揃える（幅は可変）
    resized = _resize_keep_aspect_to_h(cropped, cfg.target_h)

    # 5) max_w へ白パディング（学習と合わせる）
    w = resized.size[0]
    if w > cfg.max_w:
        resized = resized.crop((0, 0, cfg.max_w, resized.size[1]))
        w = cfg.max_w
    padded = _pad_to_max_w(resized, cfg.max_w, bg=255)

    arr = np.array(padded, dtype=np.float32) / 255.0  # [H,W]
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    valid_w_px = int(w)
    return x, valid_w_px, padded


def width_to_time_steps(valid_w_px: int) -> int:
    return max(1, int(valid_w_px) // 4)


def ctc_greedy_ids(log_probs: torch.Tensor) -> List[int]:
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
        self.root.title("Draw -> Hiragana73 CTC Inference (TorchScript)")

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

        self._last_debug_input: Optional[Image.Image] = None

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
        self._last_debug_input = None
        self.result_var.set("Cleared.")

    def infer(self):
        try:
            x, valid_w_px, dbg = preprocess_for_infer_hiragana73(
                self.img, self.cfg, self.strokes_bbox
            )
            self._last_debug_input = dbg

            x = x.to(self.device)
            with torch.no_grad():
                log_probs = self.model(x)  # [T,1,V]

            # 有効幅に対応する time steps だけに切る
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
            if self._last_debug_input is None:
                _, _, dbg = preprocess_for_infer_hiragana73(
                    self.img, self.cfg, self.strokes_bbox
                )
                self._last_debug_input = dbg

            out = Path("debug_hiragana73_ctc_input.png")
            self._last_debug_input.save(out)
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

    # must match training
    ap.add_argument("--target_h", type=int, default=32)
    ap.add_argument("--max_w", type=int, default=256)
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--ink_thresh", type=int, default=245)
    ap.add_argument("--no_invert_auto", action="store_true")

    ap.add_argument("--blur", type=float, default=0.6)

    args = ap.parse_args()

    device = "cuda" if (
        args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    cfg = InferConfig(
        canvas_size=int(args.canvas_size),
        stroke_width=int(args.stroke_width),
        target_h=int(args.target_h),
        max_w=int(args.max_w),
        pad=int(args.pad),
        ink_thresh=int(args.ink_thresh),
        invert_auto=(not args.no_invert_auto),
        blur_radius=float(args.blur),
    )

    app = DrawInferApp(Path(args.model), Path(args.vocab), cfg, device=device)
    app.run()


if __name__ == "__main__":
    main()
