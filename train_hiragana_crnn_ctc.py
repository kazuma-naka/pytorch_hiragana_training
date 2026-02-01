#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_hiragana_crnn_ctc.py
#
# CRNN + CTC trainer (supports kanji by building vocab from labels.jsonl)
#
# Key points:
# - vocab_mode=from_datasets (default): build vocab from all characters found in dataset labels (supports kanji).
# - vocab_mode=hiragana: legacy fixed hiragana vocab (+punct/space flags).
# - min_w padding: optional training-time padding to avoid tiny widths -> tiny time steps.
# - Diagnostics: shape/time-step logs, NaN/Inf checks, grad norm, val CER/exact, previews.
#
# Two-stage training (NEW):
# - Stage1: train on synthetic dataset(s)
# - Stage2: fine-tune on your own handwriting dataset(s)
# - Controlled via --two_stage and --dataset_synth/--dataset_hand

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
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
    chars: List[str] = []
    for cp in range(0x3041, 0x3097):  # ぁ..ゖ
        chars.append(chr(cp))
    if include_punct:
        chars += [
            "ー",
            "、",
            "。",
            "ペ",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ]
    if include_space:
        chars += [" "]
    out: List[str] = []
    seen = set()
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
        ids: List[int] = []
        for ch in s:
            v = self.stoi.get(ch)
            if v is None:
                raise ValueError(f"Unknown char in label: '{ch}'")
            ids.append(v)
        return ids

    def decode_greedy_ctc(self, ids: List[int]) -> str:
        out: List[str] = []
        prev: Optional[int] = None
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


def build_vocab_from_dataset_chars(
    char_freq: Counter,
    include_punct: bool,
    include_space: bool,
    min_char_freq: int = 1,
    max_vocab_chars: int = 0,
) -> Vocab:
    """
    Build vocab from dataset label characters (supports kanji).
    Deterministic ordering:
      - sort by freq desc, then codepoint asc
    """
    min_char_freq = int(min_char_freq)
    if min_char_freq < 1:
        min_char_freq = 1

    items = [(ch, n) for ch, n in char_freq.items() if n >= min_char_freq]

    # optionally force punct/space availability
    if include_punct:
        for ch in ["ー", "、", "。"]:
            if ch not in char_freq:
                items.append((ch, min_char_freq))
    if include_space:
        if " " not in char_freq:
            items.append((" ", min_char_freq))

    # de-dup then sort deterministically
    tmp: Dict[str, int] = {}
    for ch, n in items:
        tmp[ch] = max(tmp.get(ch, 0), int(n))
    items2 = list(tmp.items())
    items2.sort(key=lambda x: (-x[1], ord(x[0])))

    if max_vocab_chars and int(max_vocab_chars) > 0:
        items2 = items2[: int(max_vocab_chars)]

    chars = [ch for ch, _ in items2]
    stoi = {c: i + 1 for i, c in enumerate(chars)}  # 1.., 0 is blank
    return Vocab(itos=chars, stoi=stoi)


# ---------------- Dataset ----------------


def estimate_valid_width_from_image_L(img_L: Image.Image, ink_thresh: int = 245) -> int:
    """
    Estimate rightmost ink column (white bg, black ink). Returns valid width in pixels (>=1).
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
    return labels  # validated later


def pad_min_width_L(img_L: Image.Image, min_w: int) -> Image.Image:
    """
    Pad image (L) to at least min_w with white background on the right.
    Keeps content unchanged; valid_w should still be computed from ink (before padding).
    """
    min_w = int(min_w)
    if min_w <= 0:
        return img_L
    w, h = img_L.size
    if w >= min_w:
        return img_L
    out = Image.new("L", (min_w, h), 255)
    out.paste(img_L, (0, 0))
    return out


class MultiJsonlHandwritingDataset(Dataset):
    """
    Combine multiple datasets: each dataset is (root_dir, labels_jsonl_path).
    Each jsonl line must contain:
      - "path": relative path to image within root_dir (or a path that works under root_dir)
      - "text": label string
    """

    def __init__(
        self,
        datasets: List[Tuple[Path, Path]],
        ink_thresh: int = 245,
        min_w: int = 0,
        invert: bool = True,
    ):
        self.ink_thresh = int(ink_thresh)
        self.min_w = int(min_w)
        self.invert = bool(invert)
        self.items: List[Tuple[Path, Path, str]] = []  # (root, rel_path, text)
        self.char_freq: Counter[str] = Counter()

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
                    for ch in text:
                        self.char_freq[ch] += 1

        if not self.items:
            raise RuntimeError("No samples found across all provided datasets.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        root, rel, text = self.items[idx]
        img_path = root / rel
        img = Image.open(img_path).convert("L")  # assumed: white bg, black ink

        # valid width from ink (before padding)
        valid_w = estimate_valid_width_from_image_L(img, ink_thresh=self.ink_thresh)

        # optional: pad to minimum width to avoid tiny W -> tiny T
        if self.min_w > 0:
            img = pad_min_width_L(img, self.min_w)

        # ---- IMPORTANT: make "ink=1, bg=0" (easier for Conv+ReLU) ----
        arr = np.array(img, dtype=np.float32) / 255.0  # white=1, black=0
        if self.invert:
            arr = 1.0 - arr  # ink becomes 1, background becomes 0

        x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        return x, valid_w, text


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    val_n = max(1, int(round(n * val_ratio)))
    val = idxs[:val_n]
    train = idxs[val_n:]
    if not train:
        # keep at least 1 train sample
        train = val[1:]
        val = val[:1]
    return train, val


def collate_ctc(batch, vocab: Vocab, pad_value: float = 0.0):
    # batch: list of (x[1,H,W], valid_w, text)
    xs: List[torch.Tensor] = []
    valid_widths: List[int] = []
    targets: List[int] = []
    target_lens: List[int] = []

    for x, valid_w, text in batch:
        xs.append(x)
        valid_widths.append(int(valid_w))
        ids = vocab.encode(text)
        targets.extend(ids)
        target_lens.append(len(ids))

    max_w = max([x.shape[-1] for x in xs])
    H = xs[0].shape[1]

    # padding should match background:
    # - if invert=True => background is 0.0
    # - if invert=False => background is 1.0
    xb = torch.full((len(xs), 1, H, max_w), float(pad_value), dtype=torch.float32)
    for i, x in enumerate(xs):
        w = x.shape[-1]
        xb[i, :, :, :w] = x

    targets_t = torch.tensor(targets, dtype=torch.long)
    target_lens_t = torch.tensor(target_lens, dtype=torch.long)
    valid_widths_t = torch.tensor(valid_widths, dtype=torch.long)
    return xb, valid_widths_t, targets_t, target_lens_t


class _Subset(Dataset):
    def __init__(
        self,
        items: List[Tuple[Path, Path, str]],
        ink_thresh: int,
        min_w: int,
        invert: bool,
    ):
        self.items = items
        self.ink_thresh = int(ink_thresh)
        self.min_w = int(min_w)
        self.invert = bool(invert)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        root, rel, text = self.items[idx]
        img = Image.open(root / rel).convert("L")
        valid_w = estimate_valid_width_from_image_L(img, ink_thresh=self.ink_thresh)
        if self.min_w > 0:
            img = pad_min_width_L(img, self.min_w)

        arr = np.array(img, dtype=np.float32) / 255.0
        if self.invert:
            arr = 1.0 - arr

        x = torch.from_numpy(arr).unsqueeze(0)
        return x, valid_w, text


def build_loaders_from_items(
    *,
    items: List[Tuple[Path, Path, str]],
    vocab: Vocab,
    ink_thresh: int,
    min_w: int,
    invert: bool,
    val_ratio: float,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Build train/val loaders from a single list of items.
    Returns (train_loader, val_loader, train_n, val_n)
    """
    if not items:
        raise RuntimeError("No items to build loaders.")

    idx_train, idx_val = split_indices(len(items), val_ratio, seed)
    train_items = [items[i] for i in idx_train]
    val_items = [items[i] for i in idx_val]

    train_ds = _Subset(train_items, ink_thresh, min_w, invert=invert)
    val_ds = _Subset(val_items, ink_thresh, min_w, invert=invert)

    pad_value = 0.0 if invert else 1.0

    def collate(b):  # type: ignore
        return collate_ctc(b, vocab, pad_value=pad_value)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


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
            # keep W (so total W downsample becomes /2 instead of /4)
            nn.MaxPool2d((2, 1)),  # H:16->8, W:keep
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
    # CNN downsamples width by ~2
    return torch.clamp(valid_widths // 2, min=1)


# ---------------- Diagnostics ----------------


def _is_finite_tensor(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def _grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0
    device = parameters[0].grad.device
    total = torch.zeros((), device=device)
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total = total + (param_norm**norm_type)
    return float(total.pow(1.0 / norm_type).item())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
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


def _ctc_greedy_decode_batch(
    log_probs: torch.Tensor, input_lens: torch.Tensor, vocab: Vocab
) -> List[str]:
    # log_probs: [T,B,V]
    # input_lens: [B] (<=T)
    with torch.no_grad():
        pred_ids = torch.argmax(log_probs, dim=-1)  # [T,B]
        T, B = pred_ids.shape
        out: List[str] = []
        for b in range(B):
            tlen = int(input_lens[b].item())
            tlen = max(1, min(tlen, T))
            ids = pred_ids[:tlen, b].tolist()
            out.append(vocab.decode_greedy_ctc(ids))
        return out


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
    datasets: List[Tuple[Path, Path]]
    data_dir: Path
    labels: Path

    two_stage: bool = False
    datasets_synth: List[Tuple[Path, Path]] = None  # type: ignore
    datasets_hand: List[Tuple[Path, Path]] = None  # type: ignore

    out_dir: Path = Path("runs/hira_ctc")
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 2
    device: str = "cpu"
    val_ratio: float = 0.1

    include_punct: bool = True
    include_space: bool = True

    vocab_mode: str = "hiragana"
    min_char_freq: int = 1
    max_vocab_chars: int = 0

    rnn_hidden: int = 192
    rnn_layers: int = 2

    ink_thresh: int = 245
    min_w: int = 96
    invert: bool = True  # NEW

    log_every: int = 50
    progress_seconds: float = 2.0

    diag_every: int = 200
    preview_every: int = 1
    preview_samples: int = 8
    strict_nan: bool = False
    max_bad_ctc_ratio: float = 0.30

    stage1_epochs: int = 12
    stage1_lr: float = 3e-4
    stage2_epochs: int = 6
    stage2_lr: float = 1e-4
    save_best_by: str = "stage2"


def parse_dataset_spec(s: str) -> Tuple[Path, Path]:
    """
    ROOT:LABELS_JSONL (repeatable)
    """
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"dataset must be in 'ROOT:LABELS_JSONL' format, got: {s}"
        )
    root_s, labels_s = s.split(":", 1)
    root = Path(root_s)
    labels = Path(labels_s)

    labels2 = _resolve_labels_path(root, labels)

    if not root.is_dir():
        raise argparse.ArgumentTypeError(f"dataset root dir not found: {root}")
    if not labels2.is_file():
        raise argparse.ArgumentTypeError(f"labels jsonl not found: {labels2}")

    return root, labels2


def _print_once_header(cfg: TrainConfig, vocab: Vocab, header_extra: List[str]) -> None:
    print("========== CONFIG ==========")
    print(
        f"device={cfg.device} seed={cfg.seed} batch_size={cfg.batch_size} num_workers={cfg.num_workers}"
    )
    if cfg.two_stage:
        print(
            f"two_stage=True stage1(epochs={cfg.stage1_epochs}, lr={cfg.stage1_lr}) stage2(epochs={cfg.stage2_epochs}, lr={cfg.stage2_lr}) save_best_by={cfg.save_best_by}"
        )
    else:
        print(f"two_stage=False epochs={cfg.epochs} lr={cfg.lr}")

    print(
        f"vocab_mode={cfg.vocab_mode} vocab_size={vocab.size} (blank=0, chars={len(vocab.itos)})"
    )
    print(
        f"include_punct={cfg.include_punct} include_space={cfg.include_space} min_char_freq={cfg.min_char_freq} max_vocab_chars={cfg.max_vocab_chars}"
    )
    print(
        f"ink_thresh={cfg.ink_thresh} min_w={cfg.min_w} invert={cfg.invert} val_ratio={cfg.val_ratio}"
    )
    print(
        f"diag_every={cfg.diag_every} preview_every={cfg.preview_every} preview_samples={cfg.preview_samples} strict_nan={cfg.strict_nan} max_bad_ctc_ratio={cfg.max_bad_ctc_ratio}"
    )
    for line in header_extra:
        print(line)
    print("============================", flush=True)


def run_train_val(
    *,
    stage_name: str,
    cfg: TrainConfig,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    ctc_loss: nn.CTCLoss,
    vocab: Vocab,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pcfg: ProgressConfig,
    start_global_step: int,
    best_val_loss: float,
    ckpt_path: Path,
    save_best: bool,
) -> Tuple[int, float]:
    global_step = int(start_global_step)

    if stage_name == "stage1":
        epochs = int(cfg.stage1_epochs)
    elif stage_name == "stage2":
        epochs = int(cfg.stage2_epochs)
    else:
        epochs = int(cfg.epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        tr = EpochProgress(
            total_steps=len(train_loader),
            total_samples=max(1, len(getattr(train_loader, "dataset", []))),
            pcfg=pcfg,
        )

        for xb, valid_widths, targets, target_lens in train_loader:
            global_step += 1
            xb = xb.to(cfg.device)
            targets = targets.to(cfg.device)
            target_lens = target_lens.to(cfg.device)

            log_probs = model(xb)  # [T,B,V]
            T = int(log_probs.shape[0])

            input_lens = width_to_time_steps(valid_widths).to(cfg.device)  # [B]

            if (input_lens > T).any().item():
                input_lens = torch.clamp(input_lens, max=T)

            bad_ctc = (input_lens < target_lens).sum().item()
            bad_ctc_ratio = float(bad_ctc) / max(1.0, float(xb.shape[0]))

            loss = ctc_loss(log_probs, targets, input_lens, target_lens)

            if not torch.isfinite(loss).item():
                print(
                    f"[{stage_name} epoch {epoch}] ERROR: loss is not finite: {float(loss.item())}",
                    flush=True,
                )
                if cfg.strict_nan:
                    raise RuntimeError("Non-finite loss detected.")
            if (global_step % cfg.diag_every == 0) and (
                not _is_finite_tensor(log_probs)
            ):
                print(
                    f"[{stage_name} epoch {epoch}] ERROR: log_probs contains NaN/Inf.",
                    flush=True,
                )
                if cfg.strict_nan:
                    raise RuntimeError("Non-finite log_probs detected.")

            opt.zero_grad()
            loss.backward()
            gnorm = _grad_norm(model)
            opt.step()

            tr.update(batch_size=int(xb.shape[0]), loss_value=float(loss.item()))

            if tr.should_print():
                print(
                    tr.render(prefix=f"[{stage_name} epoch {epoch}] train:"), flush=True
                )
                tr.mark_printed()

            if global_step % cfg.diag_every == 0:
                vw = valid_widths.detach().cpu()
                il = input_lens.detach().cpu()
                tl = target_lens.detach().cpu()

                # quick sanity stats for input tensor range
                xb_cpu = xb.detach().cpu()
                xb_min = float(xb_cpu.min().item())
                xb_max = float(xb_cpu.max().item())
                xb_mean = float(xb_cpu.mean().item())

                msg: List[str] = []
                msg.append(f"[{stage_name} epoch {epoch}] DIAG step={global_step}")
                msg.append(
                    f"  xb.shape={tuple(xb.shape)}  log_probs.shape={tuple(log_probs.shape)}  T={T}"
                )
                msg.append(
                    f"  xb_stats: min={xb_min:.3f} mean={xb_mean:.3f} max={xb_max:.3f} (invert={cfg.invert})"
                )
                msg.append(
                    f"  valid_widths: min={int(vw.min())} mean={float(vw.float().mean()):.1f} max={int(vw.max())}"
                )
                msg.append(
                    f"  input_lens:   min={int(il.min())} mean={float(il.float().mean()):.1f} max={int(il.max())} (clamped<=T)"
                )
                msg.append(
                    f"  target_lens:  min={int(tl.min())} mean={float(tl.float().mean()):.2f} max={int(tl.max())}"
                )
                msg.append(
                    f"  bad_ctc(input_lens<target_lens)={int(bad_ctc)}/{int(xb.shape[0])} ({bad_ctc_ratio*100:.1f}%)"
                )
                msg.append(
                    f"  loss={float(loss.item()):.6f} grad_norm={gnorm:.3f} lr={opt.param_groups[0]['lr']:.6g}"
                )
                if bad_ctc_ratio >= cfg.max_bad_ctc_ratio:
                    msg.append(
                        "  WARN: bad_ctc_ratio is high. Consider increasing min_w, reducing downsample, or filtering tiny samples."
                    )
                print("\n".join(msg), flush=True)

        train_loss = tr.loss_sum / max(1, tr.step)

        model.eval()
        va = EpochProgress(
            total_steps=len(val_loader),
            total_samples=max(1, len(getattr(val_loader, "dataset", []))),
            pcfg=pcfg,
        )

        total_edit = 0
        total_chars = 0
        total_exact = 0
        total_samples = 0
        preview_pairs: List[Tuple[str, str]] = []

        with torch.no_grad():
            for xb, valid_widths, targets, target_lens in val_loader:
                xb = xb.to(cfg.device)
                targets = targets.to(cfg.device)
                target_lens = target_lens.to(cfg.device)

                log_probs = model(xb)
                T = int(log_probs.shape[0])

                input_lens = width_to_time_steps(valid_widths).to(cfg.device)
                if (input_lens > T).any().item():
                    input_lens = torch.clamp(input_lens, max=T)

                loss = ctc_loss(log_probs, targets, input_lens, target_lens)
                va.update(batch_size=int(xb.shape[0]), loss_value=float(loss.item()))

                if va.should_print():
                    print(
                        va.render(prefix=f"[{stage_name} epoch {epoch}]  val:"),
                        flush=True,
                    )
                    va.mark_printed()

                preds = _ctc_greedy_decode_batch(log_probs, input_lens, vocab)

                tlist = targets.detach().cpu().tolist()
                lens = target_lens.detach().cpu().tolist()
                off = 0
                for i, L in enumerate(lens):
                    ids = tlist[off : off + L]
                    off += L
                    tgt = "".join([vocab.itos[j - 1] for j in ids])
                    pred = preds[i]

                    d = _levenshtein(tgt, pred)
                    total_edit += d
                    total_chars += max(1, len(tgt))
                    total_exact += 1 if tgt == pred else 0
                    total_samples += 1

                if (epoch % cfg.preview_every == 0) and (
                    len(preview_pairs) < cfg.preview_samples
                ):
                    tlist2 = targets.detach().cpu().tolist()
                    lens2 = target_lens.detach().cpu().tolist()
                    off2 = 0
                    for i, L in enumerate(lens2):
                        if len(preview_pairs) >= cfg.preview_samples:
                            break
                        ids = tlist2[off2 : off2 + L]
                        off2 += L
                        tgt = "".join([vocab.itos[j - 1] for j in ids])
                        preview_pairs.append((tgt, preds[i]))

        val_loss = va.loss_sum / max(1, va.step)
        cer = float(total_edit) / max(1.0, float(total_chars))
        acc = float(total_exact) / max(1.0, float(total_samples))

        print(
            f"[{stage_name} epoch {epoch}] summary: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_CER={cer:.4f} val_exact={acc:.4f}",
            flush=True,
        )

        if (epoch % cfg.preview_every == 0) and preview_pairs:
            print(
                f"[{stage_name} epoch {epoch}] preview (val, greedy decode):",
                flush=True,
            )
            for i, (tgt, pred) in enumerate(
                preview_pairs[: cfg.preview_samples], start=1
            ):
                ok = "OK" if tgt == pred else "NG"
                print(f"  {i:02d} {ok}  tgt='{tgt}'  pred='{pred}'", flush=True)

        if save_best and (val_loss < best_val_loss):
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(
                f"[{stage_name} epoch {epoch}] best checkpoint saved -> {ckpt_path} (best_val_loss={best_val_loss:.6f})",
                flush=True,
            )

    return global_step, best_val_loss


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="dataset_hira")
    ap.add_argument("--labels", type=str, default="dataset_hira/labels.jsonl")

    ap.add_argument("--dataset", type=parse_dataset_spec, action="append", default=[])

    ap.add_argument("--two_stage", action="store_true")
    ap.add_argument(
        "--dataset_synth", type=parse_dataset_spec, action="append", default=[]
    )
    ap.add_argument(
        "--dataset_hand", type=parse_dataset_spec, action="append", default=[]
    )
    ap.add_argument("--stage1_epochs", type=int, default=12)
    ap.add_argument("--stage1_lr", type=float, default=3e-4)
    ap.add_argument("--stage2_epochs", type=int, default=6)
    ap.add_argument("--stage2_lr", type=float, default=1e-4)
    ap.add_argument(
        "--save_best_by", type=str, default="stage2", choices=["stage1", "stage2"]
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

    ap.add_argument(
        "--vocab_mode",
        type=str,
        default="hiragana",
        choices=["hiragana", "from_datasets"],
    )
    ap.add_argument("--min_char_freq", type=int, default=1)
    ap.add_argument("--max_vocab_chars", type=int, default=0)

    ap.add_argument("--rnn_hidden", type=int, default=192)
    ap.add_argument("--rnn_layers", type=int, default=2)

    ap.add_argument("--ink_thresh", type=int, default=245)
    ap.add_argument("--min_w", type=int, default=96)

    # NEW: invert switch
    ap.add_argument(
        "--no_invert",
        action="store_true",
        help="Disable inversion (default is invert=True => ink=1, bg=0).",
    )

    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--progress_seconds", type=float, default=2.0)

    ap.add_argument("--diag_every", type=int, default=200)
    ap.add_argument("--preview_every", type=int, default=1)
    ap.add_argument("--preview_samples", type=int, default=8)
    ap.add_argument("--strict_nan", action="store_true")
    ap.add_argument("--max_bad_ctc_ratio", type=float, default=0.30)

    args = ap.parse_args()

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    two_stage = bool(args.two_stage)
    datasets: List[Tuple[Path, Path]] = []
    datasets_synth: List[Tuple[Path, Path]] = list(args.dataset_synth or [])
    datasets_hand: List[Tuple[Path, Path]] = list(args.dataset_hand or [])

    if two_stage:
        if not datasets_synth:
            raise RuntimeError("--two_stage is set but --dataset_synth is empty.")
        if not datasets_hand:
            raise RuntimeError("--two_stage is set but --dataset_hand is empty.")
        datasets = datasets_synth + datasets_hand
    else:
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
        two_stage=two_stage,
        datasets_synth=datasets_synth if two_stage else [],
        datasets_hand=datasets_hand if two_stage else [],
        out_dir=Path(args.out_dir),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        device=device,
        val_ratio=float(args.val_ratio),
        include_punct=(not args.no_punct),
        include_space=(not args.no_space),
        vocab_mode=str(args.vocab_mode),
        min_char_freq=int(args.min_char_freq),
        max_vocab_chars=int(args.max_vocab_chars),
        rnn_hidden=int(args.rnn_hidden),
        rnn_layers=int(args.rnn_layers),
        ink_thresh=int(args.ink_thresh),
        min_w=int(args.min_w),
        invert=(not bool(args.no_invert)),
        log_every=int(args.log_every),
        progress_seconds=float(args.progress_seconds),
        diag_every=int(args.diag_every),
        preview_every=int(args.preview_every),
        preview_samples=int(args.preview_samples),
        strict_nan=bool(args.strict_nan),
        max_bad_ctc_ratio=float(args.max_bad_ctc_ratio),
        stage1_epochs=int(args.stage1_epochs),
        stage1_lr=float(args.stage1_lr),
        stage2_epochs=int(args.stage2_epochs),
        stage2_lr=float(args.stage2_lr),
        save_best_by=str(args.save_best_by),
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    ds_all = MultiJsonlHandwritingDataset(
        datasets=cfg.datasets,
        ink_thresh=cfg.ink_thresh,
        min_w=cfg.min_w,
        invert=cfg.invert,
    )

    if cfg.vocab_mode == "hiragana":
        vocab = build_vocab(
            include_punct=cfg.include_punct, include_space=cfg.include_space
        )
    else:
        vocab = build_vocab_from_dataset_chars(
            ds_all.char_freq,
            include_punct=cfg.include_punct,
            include_space=cfg.include_space,
            min_char_freq=cfg.min_char_freq,
            max_vocab_chars=cfg.max_vocab_chars,
        )
        vocab_set = set(vocab.itos)
        missing = [ch for ch in ds_all.char_freq.keys() if ch not in vocab_set]
        if missing:
            sample = missing[:50]
            raise RuntimeError(
                "Some label characters are not in vocab (likely due to min_char_freq/max_vocab_chars). "
                f"Missing sample (up to 50): {sample}. "
                "Fix by lowering --min_char_freq or increasing/removing --max_vocab_chars."
            )

    vocab_path = cfg.out_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump({"itos": vocab.itos}, f, ensure_ascii=False, indent=2)

    model = CRNNCTC(
        vocab_size=vocab.size, rnn_hidden=cfg.rnn_hidden, rnn_layers=cfg.rnn_layers
    ).to(cfg.device)
    ctc_loss = nn.CTCLoss(blank=vocab.blank, zero_infinity=True)

    ckpt = cfg.out_dir / "model_state_dict.pt"
    ts_path = cfg.out_dir / "model_torchscript.pt"

    pcfg = ProgressConfig(
        log_every=cfg.log_every, progress_seconds=cfg.progress_seconds
    )

    header_extra: List[str] = []
    header_extra.append(f"out_dir={cfg.out_dir}")

    if cfg.two_stage:
        ds_synth = MultiJsonlHandwritingDataset(
            datasets=cfg.datasets_synth,
            ink_thresh=cfg.ink_thresh,
            min_w=cfg.min_w,
            invert=cfg.invert,
        )
        ds_hand = MultiJsonlHandwritingDataset(
            datasets=cfg.datasets_hand,
            ink_thresh=cfg.ink_thresh,
            min_w=cfg.min_w,
            invert=cfg.invert,
        )

        train1, val1, tr1n, va1n = build_loaders_from_items(
            items=ds_synth.items,
            vocab=vocab,
            ink_thresh=cfg.ink_thresh,
            min_w=cfg.min_w,
            invert=cfg.invert,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        train2, val2, tr2n, va2n = build_loaders_from_items(
            items=ds_hand.items,
            vocab=vocab,
            ink_thresh=cfg.ink_thresh,
            min_w=cfg.min_w,
            invert=cfg.invert,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed + 1,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

        header_extra.append(
            f"datasets_synth={len(cfg.datasets_synth)} synth_train_samples={tr1n} synth_val_samples={va1n}"
        )
        header_extra.append(
            f"datasets_hand={len(cfg.datasets_hand)} hand_train_samples={tr2n} hand_val_samples={va2n}"
        )

        _print_once_header(cfg, vocab, header_extra=header_extra)

        best_val_loss = float("inf")
        global_step = 0

        opt1 = torch.optim.Adam(model.parameters(), lr=cfg.stage1_lr)
        save_best_stage1 = cfg.save_best_by == "stage1"
        global_step, best_val_loss = run_train_val(
            stage_name="stage1",
            cfg=cfg,
            model=model,
            opt=opt1,
            ctc_loss=ctc_loss,
            vocab=vocab,
            train_loader=train1,
            val_loader=val1,
            pcfg=pcfg,
            start_global_step=global_step,
            best_val_loss=best_val_loss,
            ckpt_path=ckpt,
            save_best=save_best_stage1,
        )

        opt2 = torch.optim.Adam(model.parameters(), lr=cfg.stage2_lr)
        save_best_stage2 = cfg.save_best_by == "stage2"
        global_step, best_val_loss = run_train_val(
            stage_name="stage2",
            cfg=cfg,
            model=model,
            opt=opt2,
            ctc_loss=ctc_loss,
            vocab=vocab,
            train_loader=train2,
            val_loader=val2,
            pcfg=pcfg,
            start_global_step=global_step,
            best_val_loss=best_val_loss,
            ckpt_path=ckpt,
            save_best=save_best_stage2,
        )

        if not ckpt.is_file():
            torch.save(model.state_dict(), ckpt)
            print(f"[final] checkpoint saved -> {ckpt}", flush=True)

    else:
        train_loader, val_loader, train_n, val_n = build_loaders_from_items(
            items=ds_all.items,
            vocab=vocab,
            ink_thresh=cfg.ink_thresh,
            min_w=cfg.min_w,
            invert=cfg.invert,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        header_extra.append(
            f"datasets={len(cfg.datasets)} train_samples={train_n} val_samples={val_n}"
        )
        _print_once_header(cfg, vocab, header_extra=header_extra)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        best_val_loss = float("inf")
        global_step = 0
        global_step, best_val_loss = run_train_val(
            stage_name="single",
            cfg=cfg,
            model=model,
            opt=opt,
            ctc_loss=ctc_loss,
            vocab=vocab,
            train_loader=train_loader,
            val_loader=val_loader,
            pcfg=pcfg,
            start_global_step=global_step,
            best_val_loss=best_val_loss,
            ckpt_path=ckpt,
            save_best=True,
        )

        if not ckpt.is_file():
            torch.save(model.state_dict(), ckpt)
            print(f"[final] checkpoint saved -> {ckpt}", flush=True)

    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model = model.to("cpu")
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(str(ts_path))

    print("Saved:", flush=True)
    print(" -", ckpt, flush=True)
    print(" -", ts_path, flush=True)
    print(" -", vocab_path, flush=True)


if __name__ == "__main__":
    main()
