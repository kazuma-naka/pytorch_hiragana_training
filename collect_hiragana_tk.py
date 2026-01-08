#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# collect_hiragana_tk.py

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import tkinter as tk
import tkinter.font as tkfont
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class CollectConfig:
    out_dir: Path
    canvas_size: int = 320
    stroke_width: int = 14
    pad: int = 20
    max_w: int = 512
    target_h: int = 32
    blur_radius: float = 0.6
    # font
    font_size: int = 12


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


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
        # crop right if too long (simple)
        return img_L.crop((0, 0, max_w, h))
    out = Image.new("L", (max_w, h), bg)
    out.paste(img_L, (0, 0))
    return out


def _tight_ink_bbox(img_L: Image.Image, ink_thresh: int = 250):
    """
    White bg(255) / black ink(0) を想定。
    ink_thresh 未満の画素を「インク」とみなし、その外接 bbox を返す。
    """
    # インク=255, 背景=0 のマスクを作る
    mask = img_L.point(lambda p: 255 if p < ink_thresh else 0, mode="L")
    return mask.getbbox()  # None if empty


def _preprocess_for_save(img_L: Image.Image, cfg: CollectConfig, bbox) -> Image.Image:
    # まず大まかに strokes_bbox で切る（任意だが高速化になる）
    cropped = _crop_with_pad(img_L, bbox, pad=0, canvas_size=cfg.canvas_size)

    # blur は「ギザギザ軽減」だが、bbox を広げる要因にもなるので
    # “余白ゼロ”が目的なら bbox 後に blur する方が良い
    # ここでは bbox→crop→blur の順にする

    # 実ピクセルからインク bbox を取り直して “余白ゼロ” crop
    tb = _tight_ink_bbox(cropped, ink_thresh=250)
    if tb is None:
        raise ValueError("Nothing drawn.")
    cropped = cropped.crop(tb)

    # ここで blur（必要なら）
    if cfg.blur_radius and cfg.blur_radius > 0:
        cropped = cropped.filter(
            ImageFilter.GaussianBlur(radius=float(cfg.blur_radius)))

    # 高さだけ揃える（幅は可変のまま）
    resized = _resize_keep_aspect_to_h(cropped, cfg.target_h)

    # 余白ゼロにしたいので padding はしない
    # max_w を超える場合だけ安全策（基本 hiragana 1文字なら超えない想定）
    if resized.size[0] > cfg.max_w:
        resized = resized.crop((0, 0, cfg.max_w, resized.size[1]))

    return resized


def pick_jp_font(root: tk.Tk, size: int) -> tkfont.Font:
    """
    Pick a Japanese-capable font if available.
    This fixes mojibake / missing glyphs in Tk widgets on many environments.
    """
    # NOTE: tkfont.families() is available after creating root.
    families = set(tkfont.families(root))

    # Prefer common JP fonts (Windows first, then Linux).
    candidates = [
        "Yu Gothic UI",
        "Yu Gothic",
        "Meiryo UI",
        "Meiryo",
        "MS Gothic",
        "MS Mincho",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "IPAGothic",
        "TakaoGothic",
        "Arial Unicode MS",
    ]

    family = next((f for f in candidates if f in families), None)
    if family:
        return tkfont.Font(root=root, family=family, size=size)

    # Fallback: Tk default (may not show JP well, but avoids crash)
    f = tkfont.nametofont("TkDefaultFont")
    f.configure(size=size)
    return f


class CollectorApp:
    def __init__(self, cfg: CollectConfig):
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.out_dir / "images").mkdir(parents=True, exist_ok=True)

        self.labels_path = self.cfg.out_dir / "labels.jsonl"

        self.root = tk.Tk()
        self.root.title("Collect Hiragana Handwriting (CTC)")

        # --- Fix Japanese text rendering in Tk widgets ---
        self.ui_font = pick_jp_font(self.root, size=self.cfg.font_size)

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
        self.strokes_bbox = None  # (minx, miny, maxx, maxy)

        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)

        tk.Label(self.root, text="Label (hiragana string):",
                 font=self.ui_font).grid(row=1, column=0, sticky="w", padx=10)

        self.label_var = tk.StringVar(value="")
        self.label_entry = tk.Entry(
            self.root,
            textvariable=self.label_var,
            width=40,
            font=self.ui_font,
        )
        self.label_entry.grid(row=1, column=1, columnspan=3,
                              sticky="ew", padx=10)

        tk.Button(self.root, text="Save", command=self.save,
                  font=self.ui_font).grid(row=1, column=4, sticky="ew", padx=10)
        tk.Button(self.root, text="Clear", command=self.clear,
                  font=self.ui_font).grid(row=1, column=5, sticky="ew", padx=10)

        self.status_var = tk.StringVar(value=f"Output: {self.cfg.out_dir}")
        tk.Label(self.root, textvariable=self.status_var,
                 font=self.ui_font).grid(row=2, column=0, columnspan=6, padx=10, pady=10)

        # Convenience: focus label entry for immediate typing
        self.label_entry.focus_set()

        self._count = 0

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
        self.label_var.set("")
        self.status_var.set("Cleared.")
        self.label_entry.focus_set()

    def clear_drawing(self):
        self.canvas.delete("all")
        self.img = Image.new(
            "L", (self.cfg.canvas_size, self.cfg.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.img)
        self.strokes_bbox = None
        self.status_var.set("Cleared drawing.")
        self.label_entry.focus_set()

    def save(self):
        text = self.label_var.get().strip()
        if not text:
            self.status_var.set("Error: label is empty.")
            self.label_entry.focus_set()
            return
        if self.strokes_bbox is None:
            self.status_var.set("Error: nothing drawn.")
            self.label_entry.focus_set()
            return

        try:
            out_img = _preprocess_for_save(
                self.img, self.cfg, self.strokes_bbox)
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            self.label_entry.focus_set()
            return

        uid = f"{_now_id()}_{self._count:06d}"
        rel = Path("images") / f"{uid}.png"
        abs_path = self.cfg.out_dir / rel
        out_img.save(abs_path)

        rec = {"path": str(rel).replace("\\", "/"), "text": text}
        with self.labels_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        self._count += 1
        self.status_var.set(f"Saved: {rel}  label='{text}'")
        self.clear_drawing()

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="dataset_hira")
    ap.add_argument("--canvas_size", type=int, default=320)
    ap.add_argument("--stroke_width", type=int, default=14)
    ap.add_argument("--pad", type=int, default=20)
    ap.add_argument("--target_h", type=int, default=32)
    ap.add_argument("--max_w", type=int, default=512)
    ap.add_argument("--blur", type=float, default=0.6)
    ap.add_argument("--font_size", type=int, default=12)
    args = ap.parse_args()

    cfg = CollectConfig(
        out_dir=Path(args.out_dir),
        canvas_size=int(args.canvas_size),
        stroke_width=int(args.stroke_width),
        pad=int(args.pad),
        target_h=int(args.target_h),
        max_w=int(args.max_w),
        blur_radius=float(args.blur),
        font_size=int(args.font_size),
    )
    app = CollectorApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
