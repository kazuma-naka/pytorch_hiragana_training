#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_hiragana_crnn_ctc.py

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------- Vocabulary ----------------

def hiragana_charset(include_punct: bool = True, include_space: bool = True) -> List[str]:
    chars = []
    for cp in range(0x3041, 0x3097):  # ぁ..ゖ
        chars.append(chr(cp))
    if include_punct:
        chars += ["ー", "、", "。"]
    if include_space:
        chars += [" "]
    out, seen = [], set()
    for c in chars:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


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


def build_vocab(include_punct: bool = True, include_space: bool = True) -> Vocab:
    chars = hiragana_charset(include_punct=include_punct,
                             include_space=include_space)
    stoi = {c: i + 1 for i, c in enumerate(chars)}  # 1.., 0 is blank
    return Vocab(itos=chars, stoi=stoi)


# ---------------- Dataset ----------------

def estimate_valid_width_from_image_L(img_L: Image.Image, ink_thresh: int = 245) -> int:
    """
    Estimate the rightmost ink column (white bg, black ink).
    Returns valid width in pixels (>=1).
    ink_thresh: pixels < ink_thresh treated as ink.
    """
    a = np.array(img_L, dtype=np.uint8)  # 0..255
    ink = a < ink_thresh
    col = ink.any(axis=0)
    idx = np.where(col)[0]
    if idx.size == 0:
        return img_L.size[0]
    return int(idx[-1]) + 1


def _resolve_labels_path(root: Path, labels: Path) -> Path:
    """
    Resolve labels path robustly:
    - If labels is absolute or exists as given -> use it
    - Else try root / labels
    """
    labels = Path(labels)
    if labels.is_file():
        return labels
    cand = root / labels
    if cand.is_file():
        return cand
    return labels  # will be validated later


class MultiJsonlHandwritingDataset(Dataset):
    """
    Combine multiple datasets: each dataset is (root_dir, labels_jsonl_path).
    Each jsonl line must contain:
      - "path": relative path to image within root_dir (or a path that works under root_dir)
      - "text": label string
    """

    def __init__(self, datasets: List[Tuple[Path, Path]], ink_thresh: int = 245):
        self.ink_thresh = int(ink_thresh)
        self.items: List[Tuple[Path, Path, str]] = []  # (root, rel_path, text)

        if not datasets:
            raise RuntimeError("No datasets provided.")

        for root, labels_jsonl in datasets:
            root = Path(root)
            labels_jsonl = _resolve_labels_path(root, Path(labels_jsonl))

            if not root.is_dir():
                raise RuntimeError(f"dataset root dir not found: {root}")
            if not labels_jsonl.is_file():
                raise RuntimeError(f"labels jsonl not found: {labels_jsonl}")

            with labels_jsonl.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception as e:
                        raise RuntimeError(
                            f"JSON parse error in {labels_jsonl} at line {line_no}: {e}"
                        )
                    rel = Path(obj["path"])
                    text = str(obj["text"])
                    self.items.append((root, rel, text))

        if not self.items:
            raise RuntimeError(
                "No samples found across all provided datasets.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        root, rel, text = self.items[idx]
        img_path = root / rel
        img = Image.open(img_path).convert("L")  # white bg, black ink

        valid_w = estimate_valid_width_from_image_L(
            img, ink_thresh=self.ink_thresh)

        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        return x, valid_w, text


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    val_n = max(1, int(round(n * val_ratio)))
    val = idxs[:val_n]
    train = idxs[val_n:]
    return train, val


def collate_ctc(batch, vocab: Vocab):
    # batch: list of (x[1,H,W], valid_w, text)
    xs = []
    valid_widths = []
    targets = []
    target_lens = []

    for x, valid_w, text in batch:
        xs.append(x)
        valid_widths.append(int(valid_w))
        ids = vocab.encode(text)
        targets.extend(ids)
        target_lens.append(len(ids))

    max_w = max([x.shape[-1] for x in xs])  # typically 512
    H = xs[0].shape[1]

    xb = torch.ones((len(xs), 1, H, max_w), dtype=torch.float32)  # white pad
    for i, x in enumerate(xs):
        w = x.shape[-1]
        xb[i, :, :, :w] = x

    targets_t = torch.tensor(targets, dtype=torch.long)
    target_lens_t = torch.tensor(target_lens, dtype=torch.long)
    valid_widths_t = torch.tensor(valid_widths, dtype=torch.long)
    return xb, valid_widths_t, targets_t, target_lens_t


# ---------------- Model (CRNN) ----------------

class ConvFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # H:32->16, W:/2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # H:16->8, W:/2 (=> W//4)
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # H:8->4, W keep
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((4, 1)),  # H:4->1, W keep
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
        f = self.feat(x)              # [B,128,1,W']
        f = f.squeeze(2)              # [B,128,W']
        f = f.permute(2, 0, 1)        # [T,B,128]
        y, _ = self.rnn(f)            # [T,B,2H]
        logits = self.fc(y)           # [T,B,V]
        return F.log_softmax(logits, dim=-1)


def width_to_time_steps(valid_widths: torch.Tensor) -> torch.Tensor:
    # CNN downsamples width by ~4
    return torch.clamp(valid_widths // 4, min=1)


# ---------------- Progress helpers ----------------

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
    # if datasets is empty, fall back to data_dir/labels
    datasets: List[Tuple[Path, Path]]
    data_dir: Path
    labels: Path

    out_dir: Path
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 2
    device: str = "cpu"
    val_ratio: float = 0.1

    include_punct: bool = True
    include_space: bool = True

    rnn_hidden: int = 192
    rnn_layers: int = 2

    ink_thresh: int = 245  # valid width detection

    # progress
    log_every: int = 50
    progress_seconds: float = 2.0


def parse_dataset_spec(s: str) -> Tuple[Path, Path]:
    """
    --dataset ROOT:LABELS_JSONL (repeatable)
    Example:
      --dataset dataset_hira:dataset_hira/labels.jsonl
      --dataset dataset_tomoe_hira:dataset_tomoe_hira/labels.jsonl
    """
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"--dataset must be in 'ROOT:LABELS_JSONL' format, got: {s}"
        )
    root_s, labels_s = s.split(":", 1)
    root = Path(root_s)
    labels = Path(labels_s)

    # If labels doesn't exist as given, try root/labels
    labels2 = _resolve_labels_path(root, labels)

    # Validate early for better UX
    if not root.is_dir():
        raise argparse.ArgumentTypeError(f"dataset root dir not found: {root}")
    if not labels2.is_file():
        raise argparse.ArgumentTypeError(f"labels jsonl not found: {labels2}")

    return root, labels2


def main():
    ap = argparse.ArgumentParser()

    # Backward compatible single-dataset args (still usable)
    ap.add_argument("--data_dir", type=str, default="dataset_hira",
                    help="(legacy) Single dataset root dir. If --dataset is provided, this is ignored.")
    ap.add_argument("--labels", type=str, default="dataset_hira/labels.jsonl",
                    help="(legacy) Single labels jsonl. If --dataset is provided, this is ignored.")

    # New multi-dataset arg
    ap.add_argument("--dataset", type=parse_dataset_spec, action="append", default=[],
                    help="Repeatable dataset spec: ROOT:LABELS_JSONL. If provided, overrides --data_dir/--labels.")

    ap.add_argument("--out_dir", type=str, default="runs/hira_ctc")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda"])
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--no_punct", action="store_true")
    ap.add_argument("--no_space", action="store_true")

    ap.add_argument("--rnn_hidden", type=int, default=192)
    ap.add_argument("--rnn_layers", type=int, default=2)

    ap.add_argument("--ink_thresh", type=int, default=245,
                    help="ink threshold for valid width detection (pixels < thresh treated as ink)")

    # progress args
    ap.add_argument("--log_every", type=int, default=50,
                    help="Print progress every N training steps (in addition to time-based printing).")
    ap.add_argument("--progress_seconds", type=float, default=2.0,
                    help="Print progress at least every N seconds.")

    args = ap.parse_args()

    # datasets: if --dataset is provided, use it; else fallback to legacy
    datasets: List[Tuple[Path, Path]] = []
    if args.dataset:
        datasets = list(args.dataset)
    else:
        root = Path(args.data_dir)
        labels = _resolve_labels_path(root, Path(args.labels))
        datasets = [(root, labels)]

    cfg = TrainConfig(
        datasets=datasets,
        data_dir=Path(args.data_dir),
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
        include_punct=(not args.no_punct),
        include_space=(not args.no_space),
        rnn_hidden=int(args.rnn_hidden),
        rnn_layers=int(args.rnn_layers),
        ink_thresh=int(args.ink_thresh),
        log_every=int(args.log_every),
        progress_seconds=float(args.progress_seconds),
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    vocab = build_vocab(include_punct=cfg.include_punct,
                        include_space=cfg.include_space)
    vocab_path = cfg.out_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump({"itos": vocab.itos}, f, ensure_ascii=False, indent=2)

    # Build combined dataset
    ds = MultiJsonlHandwritingDataset(
        datasets=cfg.datasets, ink_thresh=cfg.ink_thresh)
    train_idx, val_idx = split_indices(len(ds), cfg.val_ratio, cfg.seed)

    # Create subset views without re-reading jsonl files
    train_items = [ds.items[i] for i in train_idx]
    val_items = [ds.items[i] for i in val_idx]

    class _Subset(Dataset):
        def __init__(self, items: List[Tuple[Path, Path, str]], ink_thresh: int):
            self.items = items
            self.ink_thresh = int(ink_thresh)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx: int):
            root, rel, text = self.items[idx]
            img = Image.open(root / rel).convert("L")
            valid_w = estimate_valid_width_from_image_L(
                img, ink_thresh=self.ink_thresh)
            arr = np.array(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).unsqueeze(0)
            return x, valid_w, text

    train_ds = _Subset(train_items, cfg.ink_thresh)
    val_ds = _Subset(val_items, cfg.ink_thresh)

    def collate(b): return collate_ctc(b, vocab)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate, drop_last=False
    )

    model = CRNNCTC(vocab_size=vocab.size, rnn_hidden=cfg.rnn_hidden,
                    rnn_layers=cfg.rnn_layers).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ctc_loss = nn.CTCLoss(blank=vocab.blank, zero_infinity=True)

    best_val_loss = float("inf")
    ckpt = cfg.out_dir / "model_state_dict.pt"
    ts_path = cfg.out_dir / "model_torchscript.pt"

    pcfg = ProgressConfig(log_every=cfg.log_every,
                          progress_seconds=cfg.progress_seconds)

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

    # TorchScript export
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(str(ts_path))

    print("Saved:")
    print(" -", ckpt)
    print(" -", ts_path)
    print(" -", vocab_path)


if __name__ == "__main__":
    main()
