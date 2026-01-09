#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_hiragana73_crnn_ctc.py

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------- Download / Extract ----------------

def download_file(url: str, dst: Path, timeout: int = 60) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    dst.write_bytes(data)


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # If zip contains a single top folder, return it; else return out_dir
    children = [p for p in out_dir.iterdir() if p.name not in [".DS_Store"]]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return out_dir


# ---------------- Labels builder ----------------

_U_DIR_RE = re.compile(r"^U([0-9A-Fa-f]{4,6})$")


def char_from_udirectory_name(name: str) -> str | None:
    m = _U_DIR_RE.match(name)
    if not m:
        return None
    cp = int(m.group(1), 16)
    try:
        return chr(cp)
    except ValueError:
        return None


def list_image_files(d: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def build_labels_jsonl(dataset_root: Path, labels_jsonl: Path) -> Tuple[List[str], int]:
    """
    dataset_root contains directories like U3042, U3044, ...
    writes labels_jsonl lines: {"path": "U3042/xxx.png", "text": "あ"}
    returns (sorted unique chars, num_samples)
    """
    labels_jsonl.parent.mkdir(parents=True, exist_ok=True)

    items: List[Tuple[str, str]] = []
    chars_set = set()

    # scan only direct children that match Uxxxx
    for sub in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        ch = char_from_udirectory_name(sub.name)
        if ch is None:
            continue
        imgs = list_image_files(sub)
        if not imgs:
            continue
        chars_set.add(ch)
        for img_path in imgs:
            rel = img_path.relative_to(dataset_root).as_posix()
            items.append((rel, ch))

    if not items:
        raise RuntimeError(f"No images found under: {dataset_root}")

    with labels_jsonl.open("w", encoding="utf-8") as f:
        for rel, ch in items:
            f.write(json.dumps({"path": rel, "text": ch},
                    ensure_ascii=False) + "\n")

    chars = sorted(chars_set)
    return chars, len(items)


# ---------------- Vocab (CTC) ----------------

@dataclass
class Vocab:
    itos: List[str]          # index -> char (1..)
    stoi: Dict[str, int]     # char -> index

    @property
    def blank(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 1 + len(self.itos)  # blank + chars

    def encode(self, s: str) -> List[int]:
        ids = []
        for ch in s:
            if ch not in self.stoi:
                raise ValueError(f"Unknown char in label: '{ch}'")
            ids.append(self.stoi[ch])
        return ids

    def decode_greedy_ctc(self, ids: List[int]) -> str:
        out = []
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


def build_vocab_from_chars(chars: List[str]) -> Vocab:
    chars = list(chars)
    stoi = {c: i + 1 for i, c in enumerate(chars)}  # 1.., 0 is blank
    return Vocab(itos=chars, stoi=stoi)


# ---------------- Preprocess ----------------

def _tight_bbox_of_ink(img_L: Image.Image, ink_thresh: int) -> Tuple[int, int, int, int] | None:
    a = np.array(img_L, dtype=np.uint8)
    ink = a < ink_thresh
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
) -> Tuple[Image.Image, int]:
    """
    Returns: (processed_L_image size=(max_w,target_h), valid_width_pixels)
    - auto invert if background seems dark
    - tight crop to ink bbox (+pad)
    - resize to height=target_h (keep aspect), clamp width to max_w
    - paste at left on white canvas
    """
    img = Image.open(img_path).convert("L")

    if invert_auto:
        # If the overall image is dark, likely white ink on black bg -> invert
        m = float(np.array(img, dtype=np.uint8).mean())
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
    if h <= 0:
        h = 1
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    if new_w > max_w:
        new_w = max_w
    img = img.resize((new_w, target_h), resample=Image.Resampling.BILINEAR)

    # pad to max_w (white)
    canvas = Image.new("L", (max_w, target_h), color=255)
    canvas.paste(img, (0, 0))

    valid_w = new_w
    return canvas, valid_w


# ---------------- Dataset ----------------

class Hiragana73Dataset(Dataset):
    def __init__(
        self,
        root: Path,
        labels_jsonl: Path,
        target_h: int = 32,
        max_w: int = 256,
        pad: int = 6,
        ink_thresh: int = 245,
        invert_auto: bool = True,
    ):
        self.root = root
        self.target_h = int(target_h)
        self.max_w = int(max_w)
        self.pad = int(pad)
        self.ink_thresh = int(ink_thresh)
        self.invert_auto = bool(invert_auto)

        self.items: List[Tuple[str, str]] = []
        with labels_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.items.append((str(obj["path"]), str(obj["text"])))

        if not self.items:
            raise RuntimeError(f"No samples found in {labels_jsonl}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rel, text = self.items[idx]
        img_path = self.root / rel

        img_L, valid_w = preprocess_to_fixed_canvas(
            img_path=img_path,
            target_h=self.target_h,
            max_w=self.max_w,
            pad=self.pad,
            ink_thresh=self.ink_thresh,
            invert_auto=self.invert_auto,
        )

        arr = np.array(img_L, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W] in 0..1, white=1
        return x, int(valid_w), text


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    val_n = max(1, int(round(n * val_ratio)))
    val = idxs[:val_n]
    train = idxs[val_n:]
    return train, val


def collate_ctc(batch, vocab: Vocab):
    xs = []
    valid_widths = []
    targets = []
    target_lens = []

    for x, valid_w, text in batch:
        xs.append(x)
        valid_widths.append(int(valid_w))
        ids = vocab.encode(text)  # single char, but keep generic
        targets.extend(ids)
        target_lens.append(len(ids))

    max_w = max([x.shape[-1] for x in xs])
    H = xs[0].shape[1]

    xb = torch.ones((len(xs), 1, H, max_w), dtype=torch.float32)  # white pad
    for i, x in enumerate(xs):
        w = x.shape[-1]
        xb[i, :, :, :w] = x

    targets_t = torch.tensor(targets, dtype=torch.long)
    target_lens_t = torch.tensor(target_lens, dtype=torch.long)
    valid_widths_t = torch.tensor(valid_widths, dtype=torch.long)
    return xb, valid_widths_t, targets_t, target_lens_t


# ---------------- Model (CRNN + CTC) ----------------

class ConvFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 2)),          # H:/2, W:/2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 2)),          # H:/2, W:/2 (=> W//4)
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 1)),          # H:/2, W keep
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((4, 1)),          # H -> 1, W keep
        )

    def forward(self, x):
        return self.net(x)


class CRNNCTC(nn.Module):
    def __init__(self, vocab_size: int, rnn_hidden: int = 192, rnn_layers: int = 2):
        super().__init__()
        self.feat = ConvFeature()
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=False,  # [T,B,C]
        )
        self.fc = nn.Linear(rnn_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feat(x)        # [B,128,1,W']
        f = f.squeeze(2)        # [B,128,W']
        f = f.permute(2, 0, 1)  # [T,B,128]
        y, _ = self.rnn(f)      # [T,B,2H]
        logits = self.fc(y)     # [T,B,V]
        return F.log_softmax(logits, dim=-1)


def width_to_time_steps(valid_widths: torch.Tensor) -> torch.Tensor:
    # CNN downsamples width by ~4
    return torch.clamp(valid_widths // 4, min=1)


# ---------------- Progress ----------------

def _fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec % 60)
    h = m // 60
    m = m % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


@dataclass
class ProgressConfig:
    log_every: int = 50
    progress_seconds: float = 2.0


class EpochProgress:
    def __init__(self, total_steps: int, total_samples: int, pcfg: ProgressConfig):
        self.total_steps = max(1, int(total_steps))
        self.total_samples = max(1, int(total_samples))
        self.pcfg = pcfg

        self.start_t = time.time()
        self.last_print_t = self.start_t
        self.step = 0
        self.samples_done = 0
        self.loss_sum = 0.0

    def update(self, batch_size: int, loss_value: float) -> None:
        self.step += 1
        self.samples_done += int(batch_size)
        self.loss_sum += float(loss_value)

    def should_print(self) -> bool:
        now = time.time()
        if self.step % self.pcfg.log_every == 0:
            return True
        if (now - self.last_print_t) >= self.pcfg.progress_seconds:
            return True
        return False

    def render(self, prefix: str) -> str:
        now = time.time()
        elapsed = now - self.start_t
        steps_done = self.step
        steps_total = self.total_steps

        avg_loss = self.loss_sum / max(1, steps_done)
        sps = self.samples_done / max(1e-6, elapsed)
        step_rate = steps_done / max(1e-6, elapsed)

        remaining_steps = max(0, steps_total - steps_done)
        eta = remaining_steps / max(1e-6, step_rate)
        pct = 100.0 * steps_done / max(1, steps_total)

        return (
            f"{prefix} "
            f"{steps_done:>5d}/{steps_total:<5d} ({pct:5.1f}%) | "
            f"loss={avg_loss:.4f} | "
            f"{sps:.1f} samples/s | "
            f"elapsed={_fmt_time(elapsed)} eta={_fmt_time(eta)}"
        )

    def mark_printed(self) -> None:
        self.last_print_t = time.time()


# ---------------- Train ----------------

@dataclass
class TrainConfig:
    zip_url: str
    work_dir: Path
    dataset_dir: Path
    labels: Path
    out_dir: Path

    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 2
    device: str = "cpu"
    val_ratio: float = 0.1

    # preprocess
    target_h: int = 32
    max_w: int = 256
    pad: int = 6
    ink_thresh: int = 245
    invert_auto: bool = True

    # model
    rnn_hidden: int = 192
    rnn_layers: int = 2

    # progress
    log_every: int = 50
    progress_seconds: float = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip_url", type=str,
                    default="http://lab.ndl.go.jp/dataset/hiragana73.zip")
    ap.add_argument("--work_dir", type=str, default="ndl_cache")
    ap.add_argument("--dataset_dir", type=str,
                    default="ndl_cache/hiragana73_extracted")
    ap.add_argument("--labels", type=str,
                    default="ndl_cache/labels_hiragana73.jsonl")
    ap.add_argument("--out_dir", type=str, default="runs/hiragana73_ctc")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda"])
    ap.add_argument("--val_ratio", type=float, default=0.1)

    # preprocess
    ap.add_argument("--target_h", type=int, default=32)
    ap.add_argument("--max_w", type=int, default=256)
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--ink_thresh", type=int, default=245)
    ap.add_argument("--no_invert_auto", action="store_true")

    # model
    ap.add_argument("--rnn_hidden", type=int, default=192)
    ap.add_argument("--rnn_layers", type=int, default=2)

    # progress
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--progress_seconds", type=float, default=2.0)

    # control
    ap.add_argument("--rebuild_labels", action="store_true",
                    help="Rebuild labels.jsonl even if exists.")
    ap.add_argument("--redownload", action="store_true",
                    help="Redownload zip even if exists.")

    args = ap.parse_args()

    cfg = TrainConfig(
        zip_url=str(args.zip_url),
        work_dir=Path(args.work_dir),
        dataset_dir=Path(args.dataset_dir),
        labels=Path(args.labels),
        out_dir=Path(args.out_dir),

        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        device=("cuda" if args.device ==
                "cuda" and torch.cuda.is_available() else "cpu"),
        val_ratio=float(args.val_ratio),

        target_h=int(args.target_h),
        max_w=int(args.max_w),
        pad=int(args.pad),
        ink_thresh=int(args.ink_thresh),
        invert_auto=(not args.no_invert_auto),

        rnn_hidden=int(args.rnn_hidden),
        rnn_layers=int(args.rnn_layers),

        log_every=int(args.log_every),
        progress_seconds=float(args.progress_seconds),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.dataset_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ---- Download & Extract ----
    zip_path = cfg.work_dir / "hiragana73.zip"
    if args.redownload and zip_path.exists():
        zip_path.unlink()

    print(f"[setup] downloading: {cfg.zip_url}")
    download_file(cfg.zip_url, zip_path)

    # Extract into dataset_dir (clear only if you want; here we keep as-is)
    print(f"[setup] extracting: {zip_path} -> {cfg.dataset_dir}")
    extracted_root = extract_zip(zip_path, cfg.dataset_dir)

    # ---- Build labels ----
    if cfg.labels.exists() and (not args.rebuild_labels):
        print(
            f"[setup] labels exists: {cfg.labels} (use --rebuild_labels to regenerate)")
        # also load chars from labels
        chars_set = set()
        with cfg.labels.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                for ch in str(obj["text"]):
                    chars_set.add(ch)
        chars = sorted(chars_set)
        num_samples = sum(1 for _ in cfg.labels.open("r", encoding="utf-8"))
    else:
        print(f"[setup] building labels: root={extracted_root}")
        chars, num_samples = build_labels_jsonl(extracted_root, cfg.labels)
        print(
            f"[setup] labels written: {cfg.labels} (samples={num_samples}, classes={len(chars)})")

    # ---- Vocab ----
    vocab = build_vocab_from_chars(chars)
    vocab_path = cfg.out_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump({"itos": vocab.itos}, f, ensure_ascii=False, indent=2)

    # ---- Dataset / Split ----
    ds = Hiragana73Dataset(
        root=extracted_root,
        labels_jsonl=cfg.labels,
        target_h=cfg.target_h,
        max_w=cfg.max_w,
        pad=cfg.pad,
        ink_thresh=cfg.ink_thresh,
        invert_auto=cfg.invert_auto,
    )
    train_idx, val_idx = split_indices(len(ds), cfg.val_ratio, cfg.seed)

    class _Subset(Dataset):
        def __init__(self, base: Hiragana73Dataset, idxs: List[int]):
            self.base = base
            self.idxs = idxs

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i: int):
            return self.base[self.idxs[i]]

    train_ds = _Subset(ds, train_idx)
    val_ds = _Subset(ds, val_idx)

    def collate(b): return collate_ctc(b, vocab)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate, drop_last=False
    )

    # ---- Model ----
    model = CRNNCTC(vocab_size=vocab.size, rnn_hidden=cfg.rnn_hidden,
                    rnn_layers=cfg.rnn_layers).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ctc_loss = nn.CTCLoss(blank=vocab.blank, zero_infinity=True)

    best_val_loss = float("inf")
    ckpt = cfg.out_dir / "model_state_dict.pt"
    ts_path = cfg.out_dir / "model_torchscript.pt"

    pcfg = ProgressConfig(log_every=cfg.log_every,
                          progress_seconds=cfg.progress_seconds)

    print(
        f"[setup] device={cfg.device} samples={len(ds)} train={len(train_ds)} val={len(val_ds)} classes={len(vocab.itos)}")
    print(
        f"[setup] preprocess: target_h={cfg.target_h} max_w={cfg.max_w} pad={cfg.pad} ink_thresh={cfg.ink_thresh} invert_auto={cfg.invert_auto}")

    for epoch in range(1, cfg.epochs + 1):
        # ---- Train ----
        model.train()
        tr = EpochProgress(total_steps=len(train_loader),
                           total_samples=len(train_ds), pcfg=pcfg)

        for xb, valid_widths, targets, target_lens in train_loader:
            xb = xb.to(cfg.device)
            targets = targets.to(cfg.device)
            target_lens = target_lens.to(cfg.device)

            log_probs = model(xb)  # [T,B,V]
            input_lens = width_to_time_steps(
                valid_widths).to(cfg.device)  # [B]

            loss = ctc_loss(log_probs, targets, input_lens, target_lens)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr.update(batch_size=int(xb.shape[0]),
                      loss_value=float(loss.item()))

            if tr.should_print():
                print(tr.render(prefix=f"[epoch {epoch}] train:"), flush=True)
                tr.mark_printed()

        train_loss = tr.loss_sum / max(1, tr.step)

        # ---- Val ----
        model.eval()
        va = EpochProgress(total_steps=len(val_loader),
                           total_samples=len(val_ds), pcfg=pcfg)

        with torch.no_grad():
            for xb, valid_widths, targets, target_lens in val_loader:
                xb = xb.to(cfg.device)
                targets = targets.to(cfg.device)
                target_lens = target_lens.to(cfg.device)

                log_probs = model(xb)
                input_lens = width_to_time_steps(valid_widths).to(cfg.device)
                loss = ctc_loss(log_probs, targets, input_lens, target_lens)

                va.update(batch_size=int(
                    xb.shape[0]), loss_value=float(loss.item()))

                if va.should_print():
                    print(
                        va.render(prefix=f"[epoch {epoch}]  val:"), flush=True)
                    va.mark_printed()

        val_loss = va.loss_sum / max(1, va.step)

        print(
            f"[epoch {epoch}] summary: train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt)
            print(
                f"[epoch {epoch}] best checkpoint saved -> {ckpt}", flush=True)

    # ---- TorchScript export ----
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    # example uses configured canvas size
    example = torch.randn(1, 1, cfg.target_h, cfg.max_w)
    scripted = torch.jit.script(model)
    scripted.save(str(ts_path))

    print("Saved:")
    print(" -", ckpt)
    print(" -", ts_path)
    print(" -", vocab_path)
    print(" - labels:", cfg.labels)
    print(" - dataset_root:", extracted_root)


if __name__ == "__main__":
    main()
