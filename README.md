# Hiragana Handwriting (CRNN + CTC) — Collect / Train / Infer

## 日本語

### 概要

このリポジトリは、**手書きひらがな画像を収集し、CRNN + CTC で学習し、Tkinter 上で推論する**ための最小構成サンプルです。

- `collect_hiragana_tk.py`  
  Tkinter で手書き入力を行い、画像（PNG）とラベル（JSONL）を保存します。  
  保存時に **インク領域のみを厳密に切り出す（余白ゼロ）** 前処理を行います。

- `train_hiragana_crnn_ctc.py`  
  収集したデータ（PNG + JSONL）から CRNN-CTC を学習し、モデル（TorchScript）を出力します。

- `draw_infer_hiragana_ctc_tk_learn.py`  
  学習済み TorchScript モデルを読み込み、Tkinter で描いたひらがなを推論します。  
  前処理は `collect_hiragana_tk.py` と同一仕様に合わせています。

---

### 必要環境

- Python 3.10+（推奨: 3.12）
- 主要ライブラリ
  - `torch`
  - `numpy`
  - `Pillow`
  - `tkinter`（標準添付のことが多い）

インストール例：

```bash
pip install torch numpy pillow
```

※ Linux で tkinter が入っていない場合は、ディストリに応じて追加インストールが必要です（例: `python3-tk`）。

---

### 使い方（収集 → 学習 → 推論）

#### 1) ひらがな手書きデータ収集

```bash
python collect_hiragana_tk.py --out_dir dataset_hira
```

- 画面に描いた後、`Label` 欄に **ひらがな** を入力して `Save`
- `Clear` は全消去、保存後は自動で「描画のみクリア」します

出力：

- `dataset_hira/images/<timestamp>_<count>.png`
- `dataset_hira/labels.jsonl`

---

```bash
python3 build_dataset_from_android_pictures.py --src_root exported/Pictures/hand_writting_save_img --out_dir dataset_hira --prefix label_ --target_h 32 --max_w 512 --ink_thresh 245 --pad 0 --blur 0.0
```

```bash
python3 collect_hiragana_tk.py --out_dir dataset_hira_font --auto_elastic_alpha 18 --auto_elastic_sigma 6 --auto_scale_min 0.8 --auto_scale_max 1 --auto_erode_prob 0 --auto_edge_blur 0 --auto_elastic_prob 0 --auto_pressure_prob 0 --auto_blur 0 --auto_interval_ms 1 --auto_repeat_per_token 200 --auto_thicken_min 1 --auto_thicken_max 2 --auto_noise 0 --auto_font_ttf_list "$(printf '%s ' fonts/*.ttf fonts/*.otf 2>/dev/null)"

python3 collect_hiragana_tk.py --batch --tomoe --tomoe_tdic tomoe_data/hiragana.tdic --tomoe_per_char 200 --out_dir dataset_tomoe_hira --target_h 32 --max_w 512 --tomoe_pen_width_min 8 --tomoe_pen_width_max 16 --auto_scale_min 0.8 --auto_scale_max 1.2 --tomoe_point_jitter 0.6 --tomoe_drop_point_prob 0 --auto_pressure_prob 0 --auto_elastic_prob 0.1 --auto_elastic_alpha 6 --auto_elastic_sigma 3 --auto_erode_prob 0.3 --auto_edge_blur 0.2 --auto_blur 0.3 --auto_rotate_deg 6 --auto_noise 0.2 --tomoe_stroke_perturb_prob 0.8 --tomoe_trim_prob 0.35 --tomoe_trim_max_frac 0.18 --tomoe_extend_prob 0.25 --tomoe_extend_max_frac 0.12 --tomoe_curve_prob 0.50 --tomoe_curve_amp_px 2.2 --tomoe_curve_freq_min 0.8 --tomoe_curve_freq_max 2.0 --tomoe_perturb_resample_n 56

```

```bash
python3 collect_hiragana_tk.py --auto_elastic_alpha 18 --auto_elastic_sigma 6 --auto_scale_min 0.5 --auto_scale_max 2 --auto_erode_prob 0 --auto_edge_blur 0 --auto_elastic_prob 0 --auto_pressure_prob 0 --auto_blur 0 --auto_interval_ms 1 --auto_repeat_per_token 2000 --auto_thicken_min 2 --auto_thicken_max 4 --auto_noise 0 --auto_font_ttf_list "$(printf '%s ' fonts/*.ttf fonts/*.otf 2>/dev/null)"
```

#### 2) 学習（CRNN + CTC）

```bash
python3 train_hiragana_crnn_ctc.py   --data_dir dataset_hira   --labels dataset_hira/labels.jsonl   --out_dir runs/hira_ctc_from_dataset_hira   --epochs 200 --batch_size 32 --lr 1e-3 --device cpu
```

```bash
python3 train_hiragana_crnn_ctc.py --dataset dataset_hira:dataset_hira/labels.jsonl --out_dir runs/hira_ctc_multi --epochs 200 --batch_size 64 --lr 5e-4 --device cuda --min_w 96 --diag_every 100 --log_every 20 --preview_every 1 --preview_samples 12
```

```bash

python3 train_hiragana_crnn_ctc.py --dataset dataset_hira:dataset_hira/labels.jsonl --dataset dataset_tomoe_hira:dataset_tomoe_hira/labels.jsonl --dataset dataset_hira_font:dataset_hira_font/labels.jsonl --out_dir runs/hira_ctc_multi_fonts --epochs 200 --batch_size 32 --lr 1e-3 --device cuda --min_w 96 --diag_every 100 --log_every 20 --preview_every 1 --preview_samples 12

```

```bash
python3 preprocess_unpacked_jsonl.py --in_dir ETL4/ETL4C_unpack --out_dir dataset_etl4c_norm --white_bg yes --invert auto --autocontrast 1.0 --ink_thresh 250 --target_h 32 --max_w 512 --blur 0.6


python3 preprocess_unpacked_jsonl.py --in_dir ETL4/ETL4C_unpack --out_dir dataset_etl4c_norm --white_bg yes --include_iwiwe


```

学習のポイント：

- 入力画像は **高さ 32 に正規化**、幅は可変（最大 512 まで）
- CTC の time step は概ね `width // 4`（CNN のダウンサンプルに対応）
- `--ink_thresh`（デフォルト 245）で「インク判定（黒）」を行い有効幅を推定します
  ※ 文字が薄い/細い場合は閾値調整が有効です

生成物（`--out_dir`）：

- `model_state_dict.pt`（best checkpoint）
- `model_torchscript.pt`（推論用 TorchScript）
- `vocab.json`（文字セット）

---

#### 3) 推論（Tkinter 描画 → CTC 推論）

```bash
python3 draw_infer_hiragana_ctc_tk_learn.py --model runs/hira_ctc_from_dataset_hira/model_torchscript.pt --vocab runs/hira_ctc_from_dataset_hira/vocab.json
```

```bash
python3 draw_infer_hiragana_ctc_tk_learn.py --model runs/hira_ctc_from_dataset_tomoe_hira/model_torchscript.pt --vocab runs/hira_ctc_from_dataset_tomoe_hira/vocab.json

```

- 画面にひらがなを描いて `Infer`
- `Save debug` で、推論直前の前処理済み入力画像を `debug_hira_ctc_input.png` に保存します
  （「余白ゼロ crop が効いているか」を確認できます）

---

### データ形式

#### labels.jsonl

1 行が 1 サンプルの JSON です。

例：

```json
{ "path": "images/20260110_013000_000001.png", "text": "あ" }
```

- `path`: `dataset_hira` からの相対パス
- `text`: ひらがなラベル（複数文字も可。ただしデータ収集方針次第）

---

### 前処理仕様（重要）

保存時・推論時ともに **「余白ゼロ」前処理**を採用しています。

1. stroke の外接 bbox で粗く crop（pad=0）
2. ピクセルから「インク領域」を再計算し、tight bbox で crop
3. 必要なら blur（bbox を広げないよう crop 後に適用）
4. 高さを `target_h=32` にリサイズ（幅は可変）
5. 幅が `max_w` を超える場合のみ右側を切る（通常は 1 文字なら超えにくい）

---

### 代表的な調整パラメータ

- `--stroke_width`
  太さが変わると bbox やインク判定が変わるため、収集と推論で揃えるのが推奨です
- `--blur`
  エッジのギザギザ軽減。強くするとインク領域が太り、認識に影響する場合があります
- `--ink_thresh`
  文字が薄い場合は少し大きめ（例: 250）、背景が汚れる場合は小さめ（例: 235）

---

### トラブルシュート

- `Error: label is empty.`
  ラベル欄が空です。ひらがなラベルを入力してください。
- `Error: nothing drawn.` / `Error: Nothing drawn.`
  インク判定で bbox が取れていない可能性があります。`--ink_thresh` を上げる、線を濃く/太くする等を試してください。
- 推論が空文字になる
  学習データが不足している、または `vocab.json` の文字セットにラベル文字が含まれていない可能性があります。

---

## English

### Overview

This repository provides a minimal end-to-end pipeline to:

1. **Collect handwritten Hiragana** samples with a Tkinter canvas
2. **Train a CRNN + CTC** model on the collected dataset
3. **Run inference** in a Tkinter drawing UI using a TorchScript-exported model

Scripts:

- `collect_hiragana_tk.py`
  Draw Hiragana, type the label, and save PNG + JSONL.
  Uses **tight “no-margin” cropping** based on ink pixels.
- `train_hiragana_crnn_ctc.py`
  Trains CRNN-CTC and exports best checkpoint + TorchScript model.
- `draw_infer_hiragana_ctc_tk_learn.py`
  Loads TorchScript and runs CTC greedy decoding with matching preprocessing.

---

### Requirements

- Python 3.10+ (recommended: 3.12)
- Libraries:

  - `torch`
  - `numpy`
  - `Pillow`
  - `tkinter`

Install:

```bash
pip install torch numpy pillow
```

On some Linux distros you may need a Tk package (e.g., `python3-tk`).

---

### Usage (Collect → Train → Infer)

#### 1) Collect samples

```bash
python collect_hiragana_tk.py --out_dir dataset_hira
```

Outputs:

- `dataset_hira/images/<timestamp>_<count>.png`
- `dataset_hira/labels.jsonl`

---

#### 2) Train (CRNN + CTC)

```bash
python3 train_hiragana_crnn_ctc.py   --data_dir dataset_hira   --labels dataset_hira/labels.jsonl   --out_dir runs/hira_ctc_from_dataset_hira   --epochs 200 --batch_size 32 --lr 1e-3 --device cpu
```

```bash

python3 collect_hiragana_tk.py --batch --tomoe --tomoe_tdic tomoe_data/all.tdic --tomoe_categories hiragana,digits,kanji --tomoe_per_char 200 --out_dir dataset_tomoe_all_hira_digits_kanji --target_h 32 --max_w 512 --tomoe_pen_width_min 12 --tomoe_pen_width_max 14 --auto_scale_min 1 --auto_scale_max 1.2 --tomoe_point_jitter 0.6 --tomoe_drop_point_prob 0 --auto_pressure_prob 0 --auto_elastic_prob 0.1 --auto_elastic_alpha 6 --auto_elastic_sigma 3 --auto_erode_prob 0.3 --auto_edge_blur 0.2 --auto_blur 0.3 --auto_rotate_deg 6 --auto_noise 0.2 --tomoe_stroke_perturb_prob 0.8 --tomoe_trim_prob 0.35 --tomoe_trim_max_frac 0.18 --tomoe_extend_prob 0.25 --tomoe_extend_max_frac 0.12 --tomoe_curve_prob 0.50 --tomoe_curve_amp_px 2.2 --tomoe_curve_freq_min 0.8 --tomoe_curve_freq_max 2.0 --tomoe_perturb_resample_n 56

python3 train_hiragana_crnn_ctc.py --dataset dataset_hira:dataset_hira/labels.jsonl --dataset dataset_tomoe_hira:dataset_tomoe_hira/labels.jsonl --out_dir runs/hira_ctc_multi --epochs 300 --batch_size 32 --lr 1e-3 --device cuda --lr_scheduler plateau --plateau_patience 2 --plateau_factor 0.5 --min_lr 1e-6


python3 train_hiragana_crnn_ctc.py --dataset dataset_hira:dataset_hira/labels.jsonl --dataset dataset_tomoe_hira:dataset_tomoe_hira/labels.jsonl --out_dir runs/hira_ctc_multi --epochs 256 --batch_size 64 --lr 3e-4 --device cuda --min_w 96


python3 train_hiragana_crnn_ctc.py --dataset dataset_hira:dataset_hira/labels.jsonl --out_dir runs/hira_ctc_multi --epochs 256 --batch_size 64 --lr 5e-4 --device cuda --min_w 96 --diag_every 100 --log_every 20 --preview_every 1 --preview_samples 12

```

Artifacts in `--out_dir`:

- `model_state_dict.pt` (best checkpoint)
- `model_torchscript.pt` (TorchScript for inference)
- `vocab.json` (character set)

Notes:

- Input height is normalized to **32**, width is variable (up to 512).
- CTC time steps are roughly `width // 4` due to CNN downsampling.
- `--ink_thresh` controls ink detection for valid-width estimation.

---

#### 3) Inference (Draw → Predict)

```bash
python3 draw_infer_hiragana_ctc_tk_learn.py --model runs/hira_ctc_from_dataset_hira/model_torchscript.pt --vocab runs/hira_ctc_from_dataset_hira/vocab.json
```

- Draw Hiragana and click `Infer`
- `Save debug` saves the final preprocessed input as `debug_hira_ctc_input.png`

---

### Dataset format

#### labels.jsonl

Each line is one JSON record:

```json
{ "path": "images/20260110_013000_000001.png", "text": "あ" }
```

- `path`: relative to `dataset_hira`
- `text`: label string (single or multiple characters)

---

### Preprocessing (No-margin cropping)

Both collector and inference apply the same preprocessing:

1. Rough crop by stroke bbox (pad=0)
2. Recompute a tight ink bbox from pixel values and crop again
3. Optional blur (applied after tight crop)
4. Resize to fixed height (`target_h=32`) while keeping aspect ratio
5. If width exceeds `max_w`, crop the right side (rare for single Hiragana)

---

### Common knobs

- `--stroke_width`: keep consistent across collect/infer
- `--blur`: too much blur can thicken strokes and affect recognition
- `--ink_thresh`: increase for faint strokes; decrease for noisy backgrounds
