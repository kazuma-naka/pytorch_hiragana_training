#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# add_diacritics_to_tdic_v3.py
#
# Compose voiced/semi-voiced hiragana glyphs by adding dakuten/handakuten strokes
# to existing base glyphs in a .tdic dictionary.
#
# v3:
# - Add dakuten rotation (global + per-char override)
# - Rotation around diacritic box center
#
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

Point = Tuple[int, int]
Stroke = List[Point]
Glyph = List[Stroke]


# ---------------- TDIC IO ----------------

def parse_tdic(path: Path) -> Dict[str, Glyph]:
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    out: Dict[str, Glyph] = {}

    while i < len(lines):
        ch = lines[i].strip()
        i += 1
        if not ch:
            continue
        if i >= len(lines):
            raise ValueError(f"Unexpected EOF after char '{ch}'")

        header = lines[i].strip()
        i += 1
        if not header.startswith(":"):
            raise ValueError(f"Expected :N after '{ch}', got: {header}")
        n = int(header[1:])

        glyph: Glyph = []
        for _ in range(n):
            if i >= len(lines):
                raise ValueError(f"Unexpected EOF in strokes for '{ch}'")
            row = lines[i].strip()
            i += 1
            if not row:
                raise ValueError(f"Empty stroke line for '{ch}'")

            parts = row.split()
            k = int(parts[0])
            tokens = parts[1:]
            if len(tokens) < 2 * k:
                raise ValueError(f"Too few tokens for '{ch}': {row}")

            pts: Stroke = []
            for j in range(k):
                tx = tokens[2 * j]
                ty = tokens[2 * j + 1]
                x = int(tx.lstrip("("))
                y = int(ty.rstrip(")"))
                pts.append((x, y))

            glyph.append(pts)

        out[ch] = glyph

    return out


def write_tdic(path: Path, data: Dict[str, Glyph]) -> None:
    items = sorted(data.items(), key=lambda kv: kv[0])
    out_lines: List[str] = []
    for ch, glyph in items:
        out_lines.append(ch)
        out_lines.append(f":{len(glyph)}")
        for stroke in glyph:
            buf = [str(len(stroke))]
            for x, y in stroke:
                buf.append(f"({x} {y})")
            out_lines.append(" ".join(buf))
        out_lines.append("")
    path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")


# ---------------- Geometry ----------------

def bbox(g: Glyph) -> Tuple[int, int, int, int]:
    xs: List[int] = []
    ys: List[int] = []
    for st in g:
        for x, y in st:
            xs.append(x)
            ys.append(y)
    if not xs:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs), max(ys))


def iround(v: float) -> int:
    return int(round(v))


def rotate_point(x: float, y: float, cx: float, cy: float, deg: float) -> Tuple[float, float]:
    if abs(deg) < 1e-9:
        return (x, y)
    th = math.radians(deg)
    c = math.cos(th)
    s = math.sin(th)
    dx = x - cx
    dy = y - cy
    rx = cx + dx * c - dy * s
    ry = cy + dx * s + dy * c
    return (rx, ry)


def rotate_stroke(st: Stroke, cx: float, cy: float, deg: float) -> Stroke:
    if abs(deg) < 1e-9:
        return st
    out: Stroke = []
    for x, y in st:
        rx, ry = rotate_point(float(x), float(y), cx, cy, deg)
        out.append((iround(rx), iround(ry)))
    return out


@dataclass
class Override:
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    box_w: Optional[float] = None
    box_h: Optional[float] = None
    dakuten_style: Optional[str] = None
    rotate_deg: Optional[float] = None  # per-char dakuten rotation


@dataclass
class DiacriticConfig:
    anchor_pull: float = 0.95
    above_pull: float = 0.35

    rel_box_w: float = 0.28
    rel_box_h: float = 0.28

    min_box_w: int = 46
    min_box_h: int = 46

    dakuten_style: str = "stroke"      # "stroke" or "dots"
    dakuten_rotate_deg: float = 0.0    # global dakuten rotation (degrees)

    # two short slashes (normalized in box)
    dakuten_slash_norm = (
        ((0.18, 0.30), (0.52, 0.10)),
        ((0.48, 0.76), (0.82, 0.56)),
    )

    # dots style
    dakuten_dots_norm = (
        (0.32, 0.28),
        (0.68, 0.62),
    )
    dot_radius_norm: float = 0.06

    # handakuten
    handakuten_radius_norm: float = 0.38
    handakuten_points: int = 16
    handakuten_shift_left: float = 0.10


def load_overrides(path: Optional[Path]) -> Dict[str, Override]:
    if not path:
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("override json must be an object {char: {...}}")

    out: Dict[str, Override] = {}
    for k, v in obj.items():
        if not isinstance(v, dict):
            continue
        out[k] = Override(
            dx=float(v.get("dx", 0.0)),
            dy=float(v.get("dy", 0.0)),
            scale=float(v.get("scale", 1.0)),
            box_w=float(v["box_w"]) if "box_w" in v else None,
            box_h=float(v["box_h"]) if "box_h" in v else None,
            dakuten_style=str(v["dakuten_style"]) if "dakuten_style" in v else None,
            rotate_deg=float(v["rotate_deg"]) if "rotate_deg" in v else None,
        )
    return out


def diacritic_box(bb: Tuple[int, int, int, int], cfg: DiacriticConfig, ov: Override) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bb
    w = max(1.0, float(x1 - x0))
    h = max(1.0, float(y1 - y0))

    box_w = ov.box_w if ov.box_w is not None else max(cfg.min_box_w, w * cfg.rel_box_w)
    box_h = ov.box_h if ov.box_h is not None else max(cfg.min_box_h, h * cfg.rel_box_h)

    box_w *= ov.scale
    box_h *= ov.scale

    ax = x1 - box_w * cfg.anchor_pull
    ay = y0 - box_h * cfg.above_pull

    ax += ov.dx
    ay += ov.dy

    return (ax, ay, ax + box_w, ay + box_h)


def transform_norm_point(nx: float, ny: float, box: Tuple[float, float, float, float]) -> Point:
    bx0, by0, bx1, by1 = box
    bw = max(1e-6, bx1 - bx0)
    bh = max(1e-6, by1 - by0)
    return (iround(bx0 + nx * bw), iround(by0 + ny * bh))


# ---------------- Diacritic builders ----------------

def make_dakuten(cfg: DiacriticConfig, box: Tuple[float, float, float, float], style: str, rotate_deg: float) -> Glyph:
    style = style.lower().strip()
    if style not in ("stroke", "dots"):
        style = "stroke"

    bx0, by0, bx1, by1 = box
    cx = (bx0 + bx1) / 2.0
    cy = (by0 + by1) / 2.0

    out: Glyph = []

    if style == "stroke":
        for (p0, p1) in cfg.dakuten_slash_norm:
            st: Stroke = [
                transform_norm_point(p0[0], p0[1], box),
                transform_norm_point(p1[0], p1[1], box),
            ]
            st = rotate_stroke(st, cx, cy, rotate_deg)
            out.append(st)
        return out

    # dots: still rotate the tiny strokes so the two dots align naturally if rotated
    r = cfg.dot_radius_norm
    for (px, py) in cfg.dakuten_dots_norm:
        st: Stroke = [
            transform_norm_point(px - r, py + r, box),
            transform_norm_point(px + r, py - r, box),
        ]
        st = rotate_stroke(st, cx, cy, rotate_deg)
        out.append(st)
    return out


def make_handakuten(cfg: DiacriticConfig, box: Tuple[float, float, float, float]) -> Glyph:
    bx0, by0, bx1, by1 = box
    bw = max(1e-6, bx1 - bx0)
    bh = max(1e-6, by1 - by0)

    cx = (bx0 + bx1) / 2.0 - cfg.handakuten_shift_left * bw
    cy = (by0 + by1) / 2.0

    r = min(bw, bh) * cfg.handakuten_radius_norm
    n = max(10, cfg.handakuten_points)

    pts: Stroke = []
    for i in range(n):
        th = 2.0 * math.pi * (i / n)
        x = iround(cx + r * math.cos(th))
        y = iround(cy + r * math.sin(th))
        pts.append((x, y))
    pts.append(pts[0])

    return [pts]


# ---------------- Compose ----------------

def compose(
    data: Dict[str, Glyph],
    base: str,
    target: str,
    kind: str,
    cfg: DiacriticConfig,
    overrides: Dict[str, Override],
    overwrite: bool,
) -> bool:
    if base not in data:
        return False
    if (not overwrite) and (target in data):
        return True

    base_g = data[base]
    bb = bbox(base_g)

    ov = overrides.get(target, Override())
    box = diacritic_box(bb, cfg, ov)

    if kind == "dakuten":
        style = ov.dakuten_style or cfg.dakuten_style
        rot = ov.rotate_deg if ov.rotate_deg is not None else cfg.dakuten_rotate_deg
        dia = make_dakuten(cfg, box, style, rot)
    elif kind == "handakuten":
        dia = make_handakuten(cfg, box)
    else:
        raise ValueError(kind)

    data[target] = [st[:] for st in base_g] + dia
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tdic", required=True, type=Path)
    ap.add_argument("--out_tdic", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--include_vu", action="store_true")

    ap.add_argument("--rel_box_w", type=float, default=0.28)
    ap.add_argument("--rel_box_h", type=float, default=0.28)
    ap.add_argument("--min_box_w", type=int, default=46)
    ap.add_argument("--min_box_h", type=int, default=46)
    ap.add_argument("--anchor_pull", type=float, default=0.95)
    ap.add_argument("--above_pull", type=float, default=0.35)
    ap.add_argument("--dakuten_style", type=str, default="stroke", choices=["stroke", "dots"])
    ap.add_argument("--dakuten_rotate_deg", type=float, default=0.0)
    ap.add_argument("--handakuten_points", type=int, default=16)
    ap.add_argument("--override_json", type=Path, default=None)

    args = ap.parse_args()

    data = parse_tdic(args.in_tdic)
    cfg = DiacriticConfig(
        anchor_pull=args.anchor_pull,
        above_pull=args.above_pull,
        rel_box_w=args.rel_box_w,
        rel_box_h=args.rel_box_h,
        min_box_w=args.min_box_w,
        min_box_h=args.min_box_h,
        dakuten_style=args.dakuten_style,
        dakuten_rotate_deg=args.dakuten_rotate_deg,
        handakuten_points=args.handakuten_points,
    )
    overrides = load_overrides(args.override_json)

    dakuten_map = {
        "か": "が", "き": "ぎ", "く": "ぐ", "け": "げ", "こ": "ご",
        "さ": "ざ", "し": "じ", "す": "ず", "せ": "ぜ", "そ": "ぞ",
        "た": "だ", "ち": "ぢ", "つ": "づ", "て": "で", "と": "ど",
        "は": "ば", "ひ": "び", "ふ": "ぶ", "へ": "べ", "ほ": "ぼ",
    }
    handakuten_map = {
        "は": "ぱ", "ひ": "ぴ", "ふ": "ぷ", "へ": "ぺ", "ほ": "ぽ",
    }

    miss: List[str] = []
    added = 0

    for b, t in dakuten_map.items():
        if compose(data, b, t, "dakuten", cfg, overrides, args.overwrite):
            added += 1
        else:
            miss.append(b)

    for b, t in handakuten_map.items():
        if compose(data, b, t, "handakuten", cfg, overrides, args.overwrite):
            added += 1
        else:
            miss.append(b)

    if args.include_vu:
        if compose(data, "う", "ゔ", "dakuten", cfg, overrides, args.overwrite):
            added += 1
        else:
            miss.append("う")

    write_tdic(args.out_tdic, data)

    if miss:
        uniq = sorted(set(miss))
        print(f"[WARN] Missing base glyphs (not found in input tdic): {' '.join(uniq)}")
    print(f"[OK] wrote: {args.out_tdic} (added/updated {added} glyphs)")


if __name__ == "__main__":
    main()
