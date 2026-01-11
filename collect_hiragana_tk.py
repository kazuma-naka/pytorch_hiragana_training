#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# collect_hiragana_tk.py

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import tkinter as tk
import tkinter.font as tkfont

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk


# ---------------- Config ----------------

@dataclass
class CollectConfig:
    out_dir: Path
    canvas_size: int = 320
    stroke_width: int = 14
    pad: int = 20
    max_w: int = 512
    target_h: int = 32
    blur_radius: float = 0.6
    # ui font
    font_size: int = 12

    # --- auto draw base ---
    auto_font_ttf: Optional[Path] = None
    auto_font_px: int = 180
    auto_rotate_deg: float = 10.0
    auto_thicken: int = 2
    auto_noise: float = 0.01  # 0..1
    auto_blur: float = 0.8

    # --- randomization for learning ---
    auto_scale_min: float = 0.75
    auto_scale_max: float = 1.20
    auto_margin_px: int = 6

    # --- continuous mode ---
    auto_interval_ms: int = 250
    auto_limit: int = 0  # 0=unlimited (global limit)

    # --- sequential mode repeat per token ---
    auto_repeat_per_token: int = 1

    # initial value for GUI checkbox; runtime behavior uses the checkbox state
    auto_stop_after_sequential_cycle: bool = True

    # --- handwriting-like effects ---
    auto_elastic_alpha: float = 12.0
    auto_elastic_sigma: float = 5.0
    auto_elastic_prob: float = 0.85

    auto_pressure_prob: float = 0.95
    auto_pressure_freq_min: float = 0.6
    auto_pressure_freq_max: float = 1.6
    auto_pressure_amp_min: float = 0.25
    auto_pressure_amp_max: float = 0.65
    auto_pressure_base_min: float = 0.70
    auto_pressure_base_max: float = 1.10
    auto_dilate_min: int = 1
    auto_dilate_max: int = 5
    auto_erode_prob: float = 0.35
    auto_edge_blur: float = 0.6


# ---------------- Utilities ----------------

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


def _tight_ink_bbox(img_L: Image.Image, ink_thresh: int = 250):
    mask = img_L.point(lambda p: 255 if p < ink_thresh else 0, mode="L")
    return mask.getbbox()


def _preprocess_for_save(img_L: Image.Image, cfg: CollectConfig, bbox) -> Image.Image:
    cropped = _crop_with_pad(img_L, bbox, pad=0, canvas_size=cfg.canvas_size)

    tb = _tight_ink_bbox(cropped, ink_thresh=250)
    if tb is None:
        raise ValueError("Nothing drawn.")
    cropped = cropped.crop(tb)

    if cfg.blur_radius and cfg.blur_radius > 0:
        cropped = cropped.filter(
            ImageFilter.GaussianBlur(radius=float(cfg.blur_radius)))

    resized = _resize_keep_aspect_to_h(cropped, cfg.target_h)

    if resized.size[0] > cfg.max_w:
        resized = resized.crop((0, 0, cfg.max_w, resized.size[1]))

    return resized


# ---------------- Fonts ----------------

def pick_jp_font(root: tk.Tk, size: int) -> tkfont.Font:
    families = set(tkfont.families(root))
    candidates = [
        "Yu Gothic UI", "Yu Gothic", "Meiryo UI", "Meiryo", "MS Gothic", "MS Mincho",
        "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic", "TakaoGothic",
        "Arial Unicode MS",
    ]
    family = next((f for f in candidates if f in families), None)
    if family:
        return tkfont.Font(root=root, family=family, size=size)
    f = tkfont.nametofont("TkDefaultFont")
    f.configure(size=size)
    return f


def _guess_ttf_paths() -> List[Path]:
    candidates = [
        # Windows
        Path("C:/Windows/Fonts/YuGothM.ttc"),
        Path("C:/Windows/Fonts/YuGothB.ttc"),
        Path("C:/Windows/Fonts/meiryo.ttc"),
        Path("C:/Windows/Fonts/msgothic.ttc"),
        # Linux
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/ipaexg.ttf"),
        Path("/usr/share/fonts/truetype/ipafont-gothic/ipag.ttf"),
        Path("/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf"),
        # macOS (reference)
        Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    ]
    return [p for p in candidates if p.exists()]


def resolve_auto_font_path(cfg: CollectConfig) -> Optional[Path]:
    if cfg.auto_font_ttf is not None and Path(cfg.auto_font_ttf).exists():
        return Path(cfg.auto_font_ttf)
    for p in _guess_ttf_paths():
        try:
            ImageFont.truetype(str(p), size=16)
            return p
        except Exception:
            continue
    return None


# ---------------- Label parsing ----------------

def _parse_label_tokens(s: str) -> List[str]:
    raw = s.strip()
    if not raw:
        return []
    raw = raw.replace("、", " ").replace(",", " ")
    tokens = [t for t in (x.strip() for x in raw.split()) if t]
    if tokens:
        return tokens
    return [raw]


# ---------------- Noise / Effects ----------------

def _add_salt_noise(img_L: Image.Image, amount: float) -> Image.Image:
    if amount <= 0:
        return img_L
    w, h = img_L.size
    img = img_L.copy()
    px = img.load()
    n = int(w * h * amount)
    for _ in range(n):
        x = random.randrange(0, w)
        y = random.randrange(0, h)
        px[x, y] = random.choice([0, 30, 60, 200, 230, 255])
    return img


def _bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    wa = (x1.astype(np.float32) - x) * (y1.astype(np.float32) - y)
    wb = (x - x0.astype(np.float32)) * (y1.astype(np.float32) - y)
    wc = (x1.astype(np.float32) - x) * (y - y0.astype(np.float32))
    wd = (x - x0.astype(np.float32)) * (y - y0.astype(np.float32))

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]
    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def _smooth_random_field(h: int, w: int, sigma: float) -> np.ndarray:
    noise = np.random.uniform(0.0, 255.0, size=(h, w)).astype(np.uint8)
    im = Image.fromarray(noise, mode="L")
    if sigma and sigma > 0:
        im = im.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    arr = np.array(im).astype(np.float32)
    arr = (arr / 255.0) * 2.0 - 1.0
    return arr


def _elastic_deform_mask(mask_L: Image.Image, alpha: float, sigma: float) -> Image.Image:
    m = np.array(mask_L).astype(np.float32) / 255.0
    h, w = m.shape

    dx = _smooth_random_field(h, w, sigma) * float(alpha)
    dy = _smooth_random_field(h, w, sigma) * float(alpha)

    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    nx = xx + dx
    ny = yy + dy

    out = _bilinear_sample(m, nx, ny)
    out = np.clip(out, 0.0, 1.0)

    out_u8 = (out * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out_u8, mode="L")


def _pressure_modulate_mask(mask_L: Image.Image, cfg: CollectConfig) -> Image.Image:
    m = np.array(mask_L).astype(np.float32) / 255.0
    h, w = m.shape
    if h <= 1:
        return mask_L

    base = random.uniform(cfg.auto_pressure_base_min,
                          cfg.auto_pressure_base_max)
    amp = random.uniform(cfg.auto_pressure_amp_min, cfg.auto_pressure_amp_max)
    freq = random.uniform(cfg.auto_pressure_freq_min,
                          cfg.auto_pressure_freq_max)
    phase = random.uniform(0.0, 1.0)

    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    profile = base + amp * np.sin(2.0 * np.pi * (freq * y + phase))
    profile = np.clip(profile, 0.35, 1.65).astype(np.float32)
    profile2d = profile[:, None]

    m2 = m * profile2d

    thr = 0.45 + random.uniform(-0.08, 0.08)
    m_bin = (m2 >= thr).astype(np.uint8) * 255
    out = Image.fromarray(m_bin, mode="L")

    dil = random.randint(int(cfg.auto_dilate_min), int(cfg.auto_dilate_max))
    if dil >= 3:
        if dil % 2 == 0:
            dil += 1
        out = out.filter(ImageFilter.MaxFilter(size=dil))

    if random.random() < float(cfg.auto_erode_prob):
        er = random.choice([3, 3, 5])
        out = out.filter(ImageFilter.MinFilter(size=er))

    return out


def _mask_to_ink_image(mask_L: Image.Image, edge_blur: float) -> Image.Image:
    m = mask_L
    if edge_blur and edge_blur > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=float(edge_blur)))
        arr = np.array(m).astype(np.float32) / 255.0
        thr = 0.35 + random.uniform(-0.06, 0.06)
        m = Image.fromarray(((arr >= thr).astype(np.uint8) * 255), mode="L")

    out = Image.new("L", m.size, 255)
    out.paste(0, mask=m)
    return out


def _apply_handwriting_effects(glyph_ink_L: Image.Image, cfg: CollectConfig) -> Image.Image:
    mask = glyph_ink_L.point(lambda p: 255 if p < 200 else 0, mode="L")

    if random.random() < float(cfg.auto_pressure_prob):
        mask = _pressure_modulate_mask(mask, cfg)

    if random.random() < float(cfg.auto_elastic_prob):
        alpha = float(cfg.auto_elastic_alpha) * random.uniform(0.65, 1.15)
        sigma = float(cfg.auto_elastic_sigma) * random.uniform(0.70, 1.25)
        mask = _elastic_deform_mask(mask, alpha=alpha, sigma=sigma)

    return _mask_to_ink_image(mask, edge_blur=float(cfg.auto_edge_blur))


# ---------------- Auto draw rendering ----------------

def render_text_to_canvas_image(text: str, cfg: CollectConfig, font_path: Optional[Path]) -> Image.Image:
    canvas = Image.new("L", (cfg.canvas_size, cfg.canvas_size), 255)

    smin = min(cfg.auto_scale_min, cfg.auto_scale_max)
    smax = max(cfg.auto_scale_min, cfg.auto_scale_max)
    scale = random.uniform(smin, smax)
    font_size = max(8, int(round(cfg.auto_font_px * scale)))

    if font_path is not None:
        font = ImageFont.truetype(str(font_path), size=font_size)
    else:
        font = ImageFont.load_default()

    tmp = Image.new("L", (cfg.canvas_size, cfg.canvas_size), 255)
    td = ImageDraw.Draw(tmp)
    bbox = td.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad = 10
    gw = max(1, tw + pad * 2)
    gh = max(1, th + pad * 2)
    glyph = Image.new("L", (gw, gh), 255)
    gd = ImageDraw.Draw(glyph)

    thick = max(1, int(cfg.auto_thicken))
    base_x = pad - bbox[0]
    base_y = pad - bbox[1]
    for _ in range(thick):
        ox = random.randint(-1, 1)
        oy = random.randint(-1, 1)
        gd.text((base_x + ox, base_y + oy), text, fill=0, font=font)

    if cfg.auto_blur and cfg.auto_blur > 0:
        glyph = glyph.filter(ImageFilter.GaussianBlur(
            radius=float(cfg.auto_blur)))

    glyph = _apply_handwriting_effects(glyph, cfg)

    max_side_w = cfg.canvas_size - cfg.auto_margin_px * 2
    max_side_h = cfg.canvas_size - cfg.auto_margin_px * 2
    if glyph.size[0] > max_side_w or glyph.size[1] > max_side_h:
        sx = max_side_w / float(glyph.size[0])
        sy = max_side_h / float(glyph.size[1])
        sc = max(0.05, min(sx, sy))
        nw = max(1, int(round(glyph.size[0] * sc)))
        nh = max(1, int(round(glyph.size[1] * sc)))
        glyph = glyph.resize((nw, nh), resample=Image.Resampling.BILINEAR)

    max_x = (cfg.canvas_size - cfg.auto_margin_px) - glyph.size[0]
    max_y = (cfg.canvas_size - cfg.auto_margin_px) - glyph.size[1]
    min_x = cfg.auto_margin_px
    min_y = cfg.auto_margin_px
    if max_x < min_x:
        x = max(0, (cfg.canvas_size - glyph.size[0]) // 2)
    else:
        x = random.randint(min_x, max_x)
    if max_y < min_y:
        y = max(0, (cfg.canvas_size - glyph.size[1]) // 2)
    else:
        y = random.randint(min_y, max_y)

    canvas.paste(glyph, (x, y))

    if cfg.auto_rotate_deg and cfg.auto_rotate_deg > 0:
        deg = random.uniform(-cfg.auto_rotate_deg, cfg.auto_rotate_deg)
        canvas = canvas.rotate(
            deg,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=255
        )

    canvas = _add_salt_noise(canvas, cfg.auto_noise)

    return canvas


# ---------------- App ----------------

class CollectorApp:
    def __init__(self, cfg: CollectConfig):
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.out_dir / "images").mkdir(parents=True, exist_ok=True)

        self.labels_path = self.cfg.out_dir / "labels.jsonl"

        self.root = tk.Tk()
        self.root.title("Collect Hiragana Handwriting (CTC)")

        self.auto_font_path = resolve_auto_font_path(self.cfg)
        self.ui_font = pick_jp_font(self.root, size=self.cfg.font_size)

        self.canvas = tk.Canvas(
            self.root,
            width=cfg.canvas_size,
            height=cfg.canvas_size,
            bg="white",
            cursor="cross",
        )
        self.canvas.grid(row=0, column=0, columnspan=11, padx=10, pady=10)

        self.img = Image.new("L", (cfg.canvas_size, cfg.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.img)

        self.last: Optional[Tuple[int, int]] = None
        self.strokes_bbox = None

        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)

        # --- Controls row 1 ---
        tk.Label(self.root, text="Label list (comma/space/、):", font=self.ui_font).grid(
            row=1, column=0, sticky="w", padx=10
        )

        self.label_var = tk.StringVar(value="")
        self.label_entry = tk.Entry(
            self.root,
            textvariable=self.label_var,
            width=40,
            font=self.ui_font,
        )
        self.label_entry.grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=10)

        tk.Button(self.root, text="Save", command=self.save, font=self.ui_font).grid(
            row=1, column=4, sticky="ew", padx=6
        )
        tk.Button(self.root, text="Clear", command=self.clear, font=self.ui_font).grid(
            row=1, column=5, sticky="ew", padx=6
        )
        tk.Button(self.root, text="AutoDraw", command=self.auto_draw_and_save, font=self.ui_font).grid(
            row=1, column=6, sticky="ew", padx=6
        )
        tk.Button(self.root, text="Start", command=self.start_auto, font=self.ui_font).grid(
            row=1, column=7, sticky="ew", padx=6
        )
        tk.Button(self.root, text="Stop", command=self.stop_auto, font=self.ui_font).grid(
            row=1, column=8, sticky="ew", padx=6
        )

        tk.Label(self.root, text="Mode:", font=self.ui_font).grid(
            row=1, column=9, sticky="e", padx=(6, 2)
        )
        self.mode_var = tk.StringVar(value="random")
        self.mode_menu = tk.OptionMenu(
            self.root, self.mode_var, "random", "sequential")
        self.mode_menu.configure(font=self.ui_font)
        self.mode_menu.grid(row=1, column=10, sticky="ew", padx=(2, 10))

        # --- Controls row 2 ---
        self.preview_var = tk.BooleanVar(value=True)
        self.preview_chk = tk.Checkbutton(
            self.root,
            text="Preview",
            variable=self.preview_var,
            onvalue=True,
            offvalue=False,
            font=self.ui_font
        )
        self.preview_chk.grid(row=2, column=0, sticky="w", padx=10)

        # NEW: Stop-after-sequential checkbox (GUI toggle)
        self.stop_after_seq_var = tk.BooleanVar(
            value=bool(self.cfg.auto_stop_after_sequential_cycle))
        self.stop_after_seq_chk = tk.Checkbutton(
            self.root,
            text="Stop after sequential",
            variable=self.stop_after_seq_var,
            onvalue=True,
            offvalue=False,
            font=self.ui_font
        )
        self.stop_after_seq_chk.grid(row=2, column=1, sticky="w", padx=10)

        self.status_var = tk.StringVar(value=f"Output: {self.cfg.out_dir}")
        tk.Label(self.root, textvariable=self.status_var, font=self.ui_font).grid(
            row=2, column=2, columnspan=9, padx=10, pady=(6, 2), sticky="w"
        )

        self.current_token_var = tk.StringVar(value="Current token: (none)")
        tk.Label(self.root, textvariable=self.current_token_var, font=self.ui_font).grid(
            row=3, column=0, columnspan=11, padx=10, pady=(0, 2), sticky="w"
        )

        self.auto_count_var = tk.StringVar(value="Auto: stopped")
        tk.Label(self.root, textvariable=self.auto_count_var, font=self.ui_font).grid(
            row=4, column=0, columnspan=11, padx=10, pady=(0, 10), sticky="w"
        )

        if self.auto_font_path is None:
            self.status_var.set(
                "Warning: auto font not found. Pass --auto_font_ttf for reliable Japanese rendering."
            )

        self.label_entry.focus_set()

        self._count = 0

        self._auto_running = False
        self._auto_after_id: Optional[str] = None
        self._auto_saved = 0

        self._tokens: List[str] = []
        self._token_index: int = 0

        self._seq_current_total: int = 0
        self._seq_current_done: int = 0

        self._tk_preview: Optional[ImageTk.PhotoImage] = None
        self._canvas_preview_id: Optional[int] = None

    # ---- manual drawing ----

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
        self._clear_preview_only()
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

    def _clear_preview_only(self):
        if self._canvas_preview_id is not None:
            try:
                self.canvas.delete(self._canvas_preview_id)
            except Exception:
                pass
        self._canvas_preview_id = None
        self._tk_preview = None

    def clear(self):
        self.stop_auto()
        self.canvas.delete("all")
        self._clear_preview_only()
        self.img = Image.new(
            "L", (self.cfg.canvas_size, self.cfg.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.img)
        self.strokes_bbox = None
        self.label_var.set("")
        self.status_var.set("Cleared.")
        self.current_token_var.set("Current token: (none)")
        self.label_entry.focus_set()

    # ---- preview ----

    def show_preview(self, img_L: Image.Image):
        if not self.preview_var.get():
            return
        rgb = img_L.convert("RGB")
        self._tk_preview = ImageTk.PhotoImage(rgb)
        self.canvas.delete("all")
        self._canvas_preview_id = self.canvas.create_image(
            0, 0, anchor="nw", image=self._tk_preview)

    # ---- token selection ----

    def refresh_tokens(self) -> bool:
        tokens = _parse_label_tokens(self.label_var.get())
        if not tokens:
            return False
        self._tokens = tokens
        if self._token_index >= len(self._tokens):
            self._token_index = 0
        return True

    def _update_current_token_label(self, token: str):
        mode = self.mode_var.get().strip().lower()
        if mode == "sequential" and self._seq_current_total > 0:
            self.current_token_var.set(
                f"Current token: {token} ({self._seq_current_done}/{self._seq_current_total})")
        else:
            self.current_token_var.set(f"Current token: {token}")

    def pick_text_for_autodraw(self) -> Optional[Dict[str, Any]]:
        if not self.refresh_tokens():
            return None

        mode = self.mode_var.get().strip().lower()
        if mode != "sequential":
            token = random.choice(self._tokens)
            return {"text": token, "should_stop_after": False}

        rpt = max(1, int(self.cfg.auto_repeat_per_token))

        if self._seq_current_total <= 0:
            self._seq_current_total = rpt
            self._seq_current_done = 0

        token = self._tokens[self._token_index]
        self._seq_current_done += 1

        should_advance = (self._seq_current_done >= self._seq_current_total)

        should_stop_after = False
        if should_advance:
            is_last_token = (self._token_index == len(self._tokens) - 1)

            # IMPORTANT: runtime decision uses GUI checkbox state
            stop_after_seq = bool(self.stop_after_seq_var.get())

            if is_last_token and stop_after_seq:
                should_stop_after = True
            else:
                self._token_index += 1
                if self._token_index >= len(self._tokens):
                    self._token_index = 0
                self._seq_current_total = rpt
                self._seq_current_done = 0

        return {"text": token, "should_stop_after": should_stop_after}

    # ---- auto draw ----

    def auto_draw_and_save(self) -> bool:
        picked = self.pick_text_for_autodraw()
        if not picked:
            self.status_var.set("Error: label list is empty.")
            self.label_entry.focus_set()
            return False

        token = str(picked["text"])
        should_stop_after = bool(picked.get("should_stop_after", False))

        self._update_current_token_label(token)

        self.img = render_text_to_canvas_image(
            token, self.cfg, self.auto_font_path)
        self.draw = ImageDraw.Draw(self.img)

        tb = _tight_ink_bbox(self.img, ink_thresh=250)
        if tb is None:
            self.status_var.set("Error: nothing rendered.")
            self.label_entry.focus_set()
            return False
        self.strokes_bbox = tb

        self.show_preview(self.img)
        self.save(override_text=token)

        if should_stop_after:
            return False
        return True

    def start_auto(self):
        if self._auto_running:
            return
        if not self.refresh_tokens():
            self.status_var.set(
                "Error: label list is empty. Put labels (e.g. あ or あ,い,う) then Start."
            )
            self.label_entry.focus_set()
            return

        self._token_index = 0
        self._seq_current_total = 0
        self._seq_current_done = 0

        self._auto_running = True
        self._auto_saved = 0
        self.auto_count_var.set(
            f"Auto: running  saved={self._auto_saved}  mode={self.mode_var.get()}  interval={self.cfg.auto_interval_ms}ms  repeat={self.cfg.auto_repeat_per_token}  stop_after_seq={self.stop_after_seq_var.get()}"
        )
        self.status_var.set("Auto: started.")
        self._schedule_next_auto(immediate=True)

    def stop_auto(self):
        self._auto_running = False
        if self._auto_after_id is not None:
            try:
                self.root.after_cancel(self._auto_after_id)
            except Exception:
                pass
        self._auto_after_id = None
        self.auto_count_var.set(f"Auto: stopped  saved={self._auto_saved}")
        self.current_token_var.set("Current token: (none)")
        self.label_entry.focus_set()

    def _schedule_next_auto(self, immediate: bool = False):
        if not self._auto_running:
            return
        delay = 0 if immediate else max(1, int(self.cfg.auto_interval_ms))
        self._auto_after_id = self.root.after(delay, self._auto_tick)

    def _auto_tick(self):
        if not self._auto_running:
            return
        try:
            should_continue = self.auto_draw_and_save()
            if not should_continue:
                self.stop_auto()
                self.status_var.set("Auto: sequential completed. Stopped.")
                return

            self._auto_saved += 1
            self.auto_count_var.set(
                f"Auto: running  saved={self._auto_saved}"
                + (f" / limit={self.cfg.auto_limit}" if self.cfg.auto_limit > 0 else "")
                + f"  mode={self.mode_var.get()}  interval={self.cfg.auto_interval_ms}ms  repeat={self.cfg.auto_repeat_per_token}  stop_after_seq={self.stop_after_seq_var.get()}"
            )

            if self.cfg.auto_limit > 0 and self._auto_saved >= self.cfg.auto_limit:
                self.stop_auto()
                self.status_var.set(
                    f"Auto: reached limit={self.cfg.auto_limit}. Stopped.")
                return

        except Exception as e:
            self.stop_auto()
            self.status_var.set(f"Auto error: {e}")
            return

        self._schedule_next_auto(immediate=False)

    # ---- save ----

    def save(self, override_text: Optional[str] = None):
        text = (
            override_text if override_text is not None else self.label_var.get()).strip()
        if not text:
            self.status_var.set("Error: label is empty.")
            self.label_entry.focus_set()
            return
        if self.strokes_bbox is None:
            self.status_var.set("Error: nothing drawn.")
            self.label_entry.focus_set()
            return

        out_img = _preprocess_for_save(self.img, self.cfg, self.strokes_bbox)

        uid = f"{_now_id()}_{self._count:06d}"
        rel = Path("images") / f"{uid}.png"
        abs_path = self.cfg.out_dir / rel
        out_img.save(abs_path)

        rec = {"path": str(rel).replace("\\", "/"), "text": text}
        with self.labels_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        self._count += 1
        self.status_var.set(f"Saved: {rel}  label='{text}'")

        self.img = Image.new(
            "L", (self.cfg.canvas_size, self.cfg.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.img)
        self.strokes_bbox = None

    def run(self):
        self.root.mainloop()


# ---------------- Batch mode ----------------

def batch_generate(cfg: CollectConfig, texts: List[str], per_text: int):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "images").mkdir(parents=True, exist_ok=True)
    labels_path = cfg.out_dir / "labels.jsonl"

    font_path = resolve_auto_font_path(cfg)
    if font_path is None:
        print("Warning: auto font not found. Pass --auto_font_ttf for reliable Japanese rendering.")

    count = 0
    for t in texts:
        for _ in range(per_text):
            img = render_text_to_canvas_image(t, cfg, font_path)
            tb = _tight_ink_bbox(img, ink_thresh=250)
            if tb is None:
                continue
            out_img = _preprocess_for_save(img, cfg, tb)

            uid = f"{_now_id()}_{count:06d}"
            rel = Path("images") / f"{uid}.png"
            abs_path = cfg.out_dir / rel
            out_img.save(abs_path)

            rec = {"path": str(rel).replace("\\", "/"), "text": t}
            with labels_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            count += 1

    print(f"Done. saved={count}  out_dir={cfg.out_dir}")


# ---------------- CLI ----------------

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

    ap.add_argument("--auto_font_ttf", type=str, default="")
    ap.add_argument("--auto_font_px", type=int, default=180)
    ap.add_argument("--auto_rotate_deg", type=float, default=10.0)
    ap.add_argument("--auto_thicken", type=int, default=2)
    ap.add_argument("--auto_noise", type=float, default=0.01)
    ap.add_argument("--auto_blur", type=float, default=0.8)

    ap.add_argument("--auto_scale_min", type=float, default=0.75)
    ap.add_argument("--auto_scale_max", type=float, default=1.20)
    ap.add_argument("--auto_margin_px", type=int, default=6)

    ap.add_argument("--auto_interval_ms", type=int, default=250)
    ap.add_argument("--auto_limit", type=int, default=0)

    ap.add_argument("--auto_repeat_per_token", type=int, default=1)

    # initial checkbox state
    ap.add_argument("--auto_stop_after_sequential_cycle", action="store_true")

    ap.add_argument("--auto_elastic_alpha", type=float, default=12.0)
    ap.add_argument("--auto_elastic_sigma", type=float, default=5.0)
    ap.add_argument("--auto_elastic_prob", type=float, default=0.85)

    ap.add_argument("--auto_pressure_prob", type=float, default=0.95)
    ap.add_argument("--auto_pressure_freq_min", type=float, default=0.6)
    ap.add_argument("--auto_pressure_freq_max", type=float, default=1.6)
    ap.add_argument("--auto_pressure_amp_min", type=float, default=0.25)
    ap.add_argument("--auto_pressure_amp_max", type=float, default=0.65)
    ap.add_argument("--auto_pressure_base_min", type=float, default=0.70)
    ap.add_argument("--auto_pressure_base_max", type=float, default=1.10)
    ap.add_argument("--auto_dilate_min", type=int, default=1)
    ap.add_argument("--auto_dilate_max", type=int, default=5)
    ap.add_argument("--auto_erode_prob", type=float, default=0.35)
    ap.add_argument("--auto_edge_blur", type=float, default=0.6)

    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--texts", type=str, default="")
    ap.add_argument("--per_text", type=int, default=20)

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

        auto_font_ttf=Path(args.auto_font_ttf) if args.auto_font_ttf else None,
        auto_font_px=int(args.auto_font_px),
        auto_rotate_deg=float(args.auto_rotate_deg),
        auto_thicken=int(args.auto_thicken),
        auto_noise=float(args.auto_noise),
        auto_blur=float(args.auto_blur),

        auto_scale_min=float(args.auto_scale_min),
        auto_scale_max=float(args.auto_scale_max),
        auto_margin_px=int(args.auto_margin_px),

        auto_interval_ms=int(args.auto_interval_ms),
        auto_limit=int(args.auto_limit),

        auto_repeat_per_token=int(args.auto_repeat_per_token),

        auto_stop_after_sequential_cycle=bool(
            args.auto_stop_after_sequential_cycle),

        auto_elastic_alpha=float(args.auto_elastic_alpha),
        auto_elastic_sigma=float(args.auto_elastic_sigma),
        auto_elastic_prob=float(args.auto_elastic_prob),

        auto_pressure_prob=float(args.auto_pressure_prob),
        auto_pressure_freq_min=float(args.auto_pressure_freq_min),
        auto_pressure_freq_max=float(args.auto_pressure_freq_max),
        auto_pressure_amp_min=float(args.auto_pressure_amp_min),
        auto_pressure_amp_max=float(args.auto_pressure_amp_max),
        auto_pressure_base_min=float(args.auto_pressure_base_min),
        auto_pressure_base_max=float(args.auto_pressure_base_max),
        auto_dilate_min=int(args.auto_dilate_min),
        auto_dilate_max=int(args.auto_dilate_max),
        auto_erode_prob=float(args.auto_erode_prob),
        auto_edge_blur=float(args.auto_edge_blur),
    )

    if args.batch:
        if not args.texts.strip():
            raise SystemExit(
                "--batch requires --texts (comma/space/、-separated). e.g. --texts あ,い,う,え,お")
        tokens = _parse_label_tokens(args.texts)
        if not tokens:
            raise SystemExit("No valid texts.")
        batch_generate(cfg, texts=tokens, per_text=int(args.per_text))
        return

    app = CollectorApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
