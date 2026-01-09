```bash

python collect_hiragana_tk.py --out_dir dataset_hira

python3 train_hiragana_crnn_ctc.py   --data_dir dataset_hira   --labels dataset_hira/labels.jsonl   --out_dir runs/hira_ctc_from_dataset_hira   --epochs 200 --batch_size 32 --lr 1e-3 --device cpu

python3 draw_infer_hiragana_ctc_tk_learn.py --model runs/hira_ctc_from_dataset_hira/model_torchscript.pt --vocab runs/hira_ctc_from_dataset_hira/vocab.json

python3 train_hiragana73_crnn_ctc.py --out_dir runs/hiragana73_ctc --epochs 10 --batch_size 128 --max_w 64 --device cpu

python3 draw_infer_hiragana73_ctc_tk.py --model runs/hiragana73_ctc/model_torchscript.pt --vocab runs/hiragana73_ctc/vocab.json --max_w 256 --target_h 32 --device cpu


```
