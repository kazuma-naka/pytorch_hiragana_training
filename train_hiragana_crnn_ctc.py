#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_hiragana_crnn_ctc.py

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
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


def hiragana_charset(
    include_punct: bool = True, include_space: bool = True
) -> List[str]:
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
    itos: List[str]  # index -> char (1..)
    stoi: Dict[str, int]  # char -> index

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
    chars = hiragana_charset(include_punct=include_punct, include_space=include_space)
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
            raise RuntimeError("No samples found across all provided datasets.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        root, rel, text = self.items[idx]
        img_path = root / rel
        img = Image.open(img_path).convert("L")  # white bg, black ink

        valid_w = estimate_valid_width_from_image_L(img, ink_thresh=self.ink_thresh)

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
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # H:32->16, W:/2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # H:16->8, W:/2 (=> W//4)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # H:8->4, W keep
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
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
        f = self.feat(x)  # [B,128,1,W']
        f = f.squeeze(2)  # [B,128,W']
        f = f.permute(2, 0, 1)  # [T,B,128]
        y, _ = self.rnn(f)  # [T,B,2H]
        logits = self.fc(y)  # [T,B,V]
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


# ---------------- Metrics (CER) ----------------


def levenshtein(a: str, b: str) -> int:
    # classic DP edit distance
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # ensure b is shorter for memory efficiency
    if len(b) > len(a):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer(pred: str, gt: str) -> float:
    # character error rate
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein(pred, gt) / float(len(gt))


def greedy_decode_batch_ctc(
    log_probs: torch.Tensor, input_lens: torch.Tensor, vocab: Vocab  # [T,B,V]  # [B]
) -> List[str]:
    # argmax over vocab for each timestep
    # ids: [T,B]
    ids = torch.argmax(log_probs, dim=-1).detach().cpu().numpy()
    lens = input_lens.detach().cpu().numpy().astype(np.int64)

    out: List[str] = []
    T, B = ids.shape
    for b in range(B):
        tlen = int(lens[b])
        tlen = max(1, min(int(tlen), int(T)))
        seq = ids[:tlen, b].tolist()
        out.append(vocab.decode_greedy_ctc(seq))
    return out


# ---------------- Logging ----------------


def setup_logger(out_dir: Path, log_file: Optional[str] = None) -> logging.Logger:
    """
    Logger:
    - Always logs to stdout.
    - Logs to file ONLY if log_file is not None / not empty.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("hira_ctc_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        log_path = out_dir / log_file
        fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class MetricsWriter:
    """
    Metrics files are created ONLY when this class is instantiated.
    (So: do NOT instantiate unless user explicitly requested metrics output.)
    """

    def __init__(self, out_dir: Path):
        self.jsonl_path = out_dir / "metrics.jsonl"
        self.csv_path = out_dir / "metrics.csv"
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = None

        # create/clear jsonl
        self.jsonl_path.write_text("", encoding="utf-8")

    def write(self, row: Dict) -> None:
        # jsonl
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # csv
        if self._csv_writer is None:
            fieldnames = list(row.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        try:
            self._csv_file.close()
        except Exception:
            pass


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

    # evaluation
    eval_samples: int = (
        256  # number of val samples used to estimate CER (speed vs accuracy)
    )

    # lr scheduler
    lr_scheduler: str = "none"  # none | plateau
    plateau_mode: str = "min"  # min|max
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    plateau_threshold: float = 1e-4
    plateau_threshold_mode: str = "rel"  # rel|abs
    plateau_cooldown: int = 0
    min_lr: float = 1e-6

    # output controls (NEW)
    log_file: Optional[str] = None  # e.g. "train.log" (default: None -> no file)
    write_metrics: bool = False  # default: no metrics.jsonl/csv


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


def _get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    lrs = []
    for pg in optimizer.param_groups:
        if "lr" in pg:
            lrs.append(float(pg["lr"]))
    if not lrs:
        return float("nan")
    # In most cases, all param groups share same lr. If not, we log the first.
    return float(lrs[0])


def _normalize_optional_str(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip()
    if t == "":
        return None
    if t.lower() in ("none", "null", "no", "false", "0"):
        return None
    return t


def main():
    ap = argparse.ArgumentParser()

    # Backward compatible single-dataset args (still usable)
    ap.add_argument(
        "--data_dir",
        type=str,
        default="dataset_hira",
        help="(legacy) Single dataset root dir. If --dataset is provided, this is ignored.",
    )
    ap.add_argument(
        "--labels",
        type=str,
        default="dataset_hira/labels.jsonl",
        help="(legacy) Single labels jsonl. If --dataset is provided, this is ignored.",
    )

    # New multi-dataset arg
    ap.add_argument(
        "--dataset",
        type=parse_dataset_spec,
        action="append",
        default=[],
        help="Repeatable dataset spec: ROOT:LABELS_JSONL. If provided, overrides --data_dir/--labels.",
    )

    ap.add_argument("--out_dir", type=str, default="runs/hira_ctc")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--no_punct", action="store_true")
    ap.add_argument("--no_space", action="store_true")

    ap.add_argument("--rnn_hidden", type=int, default=192)
    ap.add_argument("--rnn_layers", type=int, default=2)

    ap.add_argument(
        "--ink_thresh",
        type=int,
        default=245,
        help="ink threshold for valid width detection (pixels < thresh treated as ink)",
    )

    # progress args
    ap.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Print progress every N training steps (in addition to time-based printing).",
    )
    ap.add_argument(
        "--progress_seconds",
        type=float,
        default=2.0,
        help="Print progress at least every N seconds.",
    )

    # eval args
    ap.add_argument(
        "--eval_samples",
        type=int,
        default=256,
        help="How many validation samples to use for CER estimation each epoch (0 disables CER).",
    )

    # lr scheduler args
    ap.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=["none", "plateau"],
        help="Learning-rate scheduler. 'plateau' reduces LR when val_loss stops improving.",
    )
    ap.add_argument(
        "--plateau_mode",
        type=str,
        default="min",
        choices=["min", "max"],
        help="ReduceLROnPlateau mode. Use 'min' for val_loss.",
    )
    ap.add_argument(
        "--plateau_factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor. NewLR = LR * factor when triggered.",
    )
    ap.add_argument(
        "--plateau_patience",
        type=int,
        default=2,
        help="ReduceLROnPlateau patience (epochs with no improvement before reducing).",
    )
    ap.add_argument(
        "--plateau_threshold",
        type=float,
        default=1e-4,
        help="ReduceLROnPlateau threshold for measuring new optimum.",
    )
    ap.add_argument(
        "--plateau_threshold_mode",
        type=str,
        default="rel",
        choices=["rel", "abs"],
        help="ReduceLROnPlateau threshold_mode.",
    )
    ap.add_argument(
        "--plateau_cooldown",
        type=int,
        default=0,
        help="ReduceLROnPlateau cooldown (epochs to wait after LR reduction).",
    )
    ap.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum LR for scheduler (lower bound).",
    )

    # NEW: output controls
    ap.add_argument(
        "--log_file",
        type=str,
        default="",
        help="If set, also write logs to this file under out_dir (e.g. 'train.log'). Default: disabled.",
    )
    ap.add_argument(
        "--write_metrics",
        action="store_true",
        help="If set, write metrics.jsonl and metrics.csv under out_dir. Default: disabled.",
    )

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
        device=(
            "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        ),
        val_ratio=float(args.val_ratio),
        include_punct=(not args.no_punct),
        include_space=(not args.no_space),
        rnn_hidden=int(args.rnn_hidden),
        rnn_layers=int(args.rnn_layers),
        ink_thresh=int(args.ink_thresh),
        log_every=int(args.log_every),
        progress_seconds=float(args.progress_seconds),
        eval_samples=int(args.eval_samples),
        lr_scheduler=str(args.lr_scheduler),
        plateau_mode=str(args.plateau_mode),
        plateau_factor=float(args.plateau_factor),
        plateau_patience=int(args.plateau_patience),
        plateau_threshold=float(args.plateau_threshold),
        plateau_threshold_mode=str(args.plateau_threshold_mode),
        plateau_cooldown=int(args.plateau_cooldown),
        min_lr=float(args.min_lr),
        log_file=_normalize_optional_str(args.log_file),
        write_metrics=bool(args.write_metrics),
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # logger: file output disabled by default
    logger = setup_logger(cfg.out_dir, log_file=cfg.log_file)

    # metrics: disabled by default (so metrics.jsonl/csv are NOT created unless requested)
    metrics: Optional[MetricsWriter] = (
        MetricsWriter(cfg.out_dir) if cfg.write_metrics else None
    )

    logger.info(
        "Config: %s",
        json.dumps(
            {
                "datasets": [(str(r), str(l)) for (r, l) in cfg.datasets],
                "out_dir": str(cfg.out_dir),
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "seed": cfg.seed,
                "num_workers": cfg.num_workers,
                "device": cfg.device,
                "val_ratio": cfg.val_ratio,
                "include_punct": cfg.include_punct,
                "include_space": cfg.include_space,
                "rnn_hidden": cfg.rnn_hidden,
                "rnn_layers": cfg.rnn_layers,
                "ink_thresh": cfg.ink_thresh,
                "log_every": cfg.log_every,
                "progress_seconds": cfg.progress_seconds,
                "eval_samples": cfg.eval_samples,
                "lr_scheduler": cfg.lr_scheduler,
                "plateau_mode": cfg.plateau_mode,
                "plateau_factor": cfg.plateau_factor,
                "plateau_patience": cfg.plateau_patience,
                "plateau_threshold": cfg.plateau_threshold,
                "plateau_threshold_mode": cfg.plateau_threshold_mode,
                "plateau_cooldown": cfg.plateau_cooldown,
                "min_lr": cfg.min_lr,
                "log_file": cfg.log_file,
                "write_metrics": cfg.write_metrics,
            },
            ensure_ascii=False,
        ),
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    vocab = build_vocab(
        include_punct=cfg.include_punct, include_space=cfg.include_space
    )
    vocab_path = cfg.out_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump({"itos": vocab.itos}, f, ensure_ascii=False, indent=2)

    # Build combined dataset
    ds = MultiJsonlHandwritingDataset(datasets=cfg.datasets, ink_thresh=cfg.ink_thresh)
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
            valid_w = estimate_valid_width_from_image_L(img, ink_thresh=self.ink_thresh)
            arr = np.array(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).unsqueeze(0)
            return x, valid_w, text

    train_ds = _Subset(train_items, cfg.ink_thresh)
    val_ds = _Subset(val_items, cfg.ink_thresh)

    def collate(b):
        return collate_ctc(b, vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        drop_last=False,
    )

    model = CRNNCTC(
        vocab_size=vocab.size, rnn_hidden=cfg.rnn_hidden, rnn_layers=cfg.rnn_layers
    ).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ctc_loss = nn.CTCLoss(blank=vocab.blank, zero_infinity=True)

    # LR Scheduler
    scheduler = None
    if cfg.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=cfg.plateau_mode,
            factor=float(cfg.plateau_factor),
            patience=int(cfg.plateau_patience),
            threshold=float(cfg.plateau_threshold),
            threshold_mode=cfg.plateau_threshold_mode,
            cooldown=int(cfg.plateau_cooldown),
            min_lr=float(cfg.min_lr),
        )
        logger.info(
            "LR Scheduler: ReduceLROnPlateau(mode=%s, factor=%s, patience=%s, threshold=%s, threshold_mode=%s, cooldown=%s, min_lr=%s)",
            cfg.plateau_mode,
            str(cfg.plateau_factor),
            str(cfg.plateau_patience),
            str(cfg.plateau_threshold),
            cfg.plateau_threshold_mode,
            str(cfg.plateau_cooldown),
            str(cfg.min_lr),
        )
    else:
        logger.info("LR Scheduler: none")

    best_val_loss = float("inf")
    best_val_cer = float("inf")
    ckpt = cfg.out_dir / "model_state_dict.pt"
    ts_path = cfg.out_dir / "model_torchscript.pt"

    pcfg = ProgressConfig(
        log_every=cfg.log_every, progress_seconds=cfg.progress_seconds
    )

    t0_all = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_t0 = time.time()

        # ---- Train ----
        model.train()
        tr = EpochProgress(
            total_steps=len(train_loader), total_samples=len(train_ds), pcfg=pcfg
        )

        for xb, valid_widths, targets, target_lens in train_loader:
            xb = xb.to(cfg.device)
            targets = targets.to(cfg.device)
            target_lens = target_lens.to(cfg.device)

            log_probs = model(xb)  # [T,B,V]
            input_lens = width_to_time_steps(valid_widths).to(cfg.device)  # [B]

            loss = ctc_loss(log_probs, targets, input_lens, target_lens)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr.update(batch_size=int(xb.shape[0]), loss_value=float(loss.item()))

            if tr.should_print():
                logger.info(tr.render(prefix=f"[epoch {epoch}] train:"))
                tr.mark_printed()

        train_loss = tr.loss_sum / max(1, tr.step)

        # ---- Val ----
        model.eval()
        va = EpochProgress(
            total_steps=len(val_loader), total_samples=len(val_ds), pcfg=pcfg
        )
        with torch.no_grad():
            for xb, valid_widths, targets, target_lens in val_loader:
                xb = xb.to(cfg.device)
                targets = targets.to(cfg.device)
                target_lens = target_lens.to(cfg.device)

                log_probs = model(xb)
                input_lens = width_to_time_steps(valid_widths).to(cfg.device)
                loss = ctc_loss(log_probs, targets, input_lens, target_lens)

                va.update(batch_size=int(xb.shape[0]), loss_value=float(loss.item()))

                if va.should_print():
                    logger.info(va.render(prefix=f"[epoch {epoch}]  val:"))
                    va.mark_printed()

        val_loss = va.loss_sum / max(1, va.step)

        # ---- CER computation (lightweight pass) ----
        val_cer = None
        if cfg.eval_samples > 0:
            model.eval()
            need = int(cfg.eval_samples)
            got = 0
            cer_sum2 = 0.0

            with torch.no_grad():
                for i in range(0, len(val_ds)):
                    if got >= need:
                        break
                    x, valid_w, gt = val_ds[i]
                    xb1 = x.unsqueeze(0).to(cfg.device)  # [1,1,H,W]
                    log_probs = model(xb1)  # [T,1,V]
                    input_lens = width_to_time_steps(
                        torch.tensor([int(valid_w)], dtype=torch.long)
                    ).to(cfg.device)
                    pred = greedy_decode_batch_ctc(log_probs, input_lens, vocab)[0]
                    cer_sum2 += cer(pred, gt)
                    got += 1

            val_cer = cer_sum2 / max(1, got)

        # ---- LR Scheduler step (after val) ----
        lr_before = _get_current_lr(opt)
        if scheduler is not None:
            # plateau uses validation metric
            scheduler.step(val_loss)
        lr_after = _get_current_lr(opt)
        if lr_after != lr_before:
            logger.info(
                "[epoch %d] LR updated: %.8f -> %.8f",
                epoch,
                float(lr_before),
                float(lr_after),
            )

        epoch_sec = time.time() - epoch_t0
        all_sec = time.time() - t0_all

        # summary
        if val_cer is None:
            logger.info(
                "[epoch %d] summary: train_loss=%.4f val_loss=%.4f lr=%.8f epoch=%s total=%s",
                epoch,
                train_loss,
                val_loss,
                float(lr_after),
                _fmt_time(epoch_sec),
                _fmt_time(all_sec),
            )
        else:
            logger.info(
                "[epoch %d] summary: train_loss=%.4f val_loss=%.4f val_CER=%.4f lr=%.8f epoch=%s total=%s",
                epoch,
                train_loss,
                val_loss,
                float(val_cer),
                float(lr_after),
                _fmt_time(epoch_sec),
                _fmt_time(all_sec),
            )

        # write metrics row ONLY if enabled
        if metrics is not None:
            row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_cer": None if val_cer is None else float(val_cer),
                "lr": float(lr_after),
                "batch_size": int(cfg.batch_size),
                "device": str(cfg.device),
                "train_samples": int(len(train_ds)),
                "val_samples": int(len(val_ds)),
                "epoch_seconds": float(epoch_sec),
                "total_seconds": float(all_sec),
                "lr_scheduler": str(cfg.lr_scheduler),
            }
            metrics.write(row)

        # checkpoint
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt)
            logger.info(
                "[epoch %d] best checkpoint saved -> %s (best_val_loss=%.6f)",
                epoch,
                str(ckpt),
                float(best_val_loss),
            )

        if val_cer is not None and float(val_cer) < best_val_cer:
            best_val_cer = float(val_cer)
            logger.info(
                "[epoch %d] new best val_CER -> %.6f", epoch, float(best_val_cer)
            )

    # TorchScript export
    logger.info("Export TorchScript...")

    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)

    model = model.to("cpu")  # 必須
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save(str(ts_path))

    if metrics is not None:
        metrics.close()

    logger.info("Saved:")
    logger.info(" - %s", str(ckpt))
    logger.info(" - %s", str(ts_path))
    logger.info(" - %s", str(vocab_path))
    if cfg.log_file:
        logger.info(" - %s", str(cfg.out_dir / cfg.log_file))
    if cfg.write_metrics:
        logger.info(" - %s", str(cfg.out_dir / "metrics.jsonl"))
        logger.info(" - %s", str(cfg.out_dir / "metrics.csv"))


if __name__ == "__main__":
    main()
