---
license: mit
datasets:
- Helsinki-NLP/open_subtitles
language:
- zh
base_model:
- hfl/chinese-macbert-base
pipeline_tag: text-classification
tags:
- agent
- nlp
- chinese
- sentiment-analysis
- emotion
- regression
- vad
- valence-arousal-dominance
- transformers
- bert
- macbert
---

# vad-macbert

Chinese VAD (valence/arousal/dominance) regression based on `hfl/chinese-macbert-base`.
The model predicts 3 continuous values aligned to the VAD scale produced by
`RobroKools/vad-bert` (teacher model).

## Quickstart

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "Pectics/vad-macbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

text = "这部电影让我很感动。"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
vad = outputs.logits.squeeze().tolist()
print("VAD:", vad)
```

## Model Details

- Base model: `hfl/chinese-macbert-base`
- Task: VAD regression (3 outputs: valence, arousal, dominance)
- Head: `AutoModelForSequenceClassification` with `num_labels=3`, `problem_type=regression`

## Data Sources & Labeling

### en-zh_cn_vad_clean.csv
- Source: OpenSubtitles EN-ZH parallel corpus.
- Labeling: English side fed into `RobroKools/vad-bert` to obtain VAD values,
  then assigned to the paired Chinese text.

### en-zh_cn_vad_long.csv
- Derived from `en-zh_cn_vad_clean.csv` by filtering for longer texts using a
  length threshold (original threshold was not recorded).
- Inferred from statistics: minimum length is 32 characters, so the filter
  likely kept samples with length >= 32 chars.

### en-zh_cn_vad_long_clean.csv
- Cleaned from `en-zh_cn_vad_long.csv` by removing subtitle formatting noise:
  - ASS/SSA tag blocks like `{\\fs..\\pos(..)}` (including broken `{` blocks)
  - HTML-like tags (e.g. `<i>...</i>`)
  - Escape codes like `\\N`, `\\n`, `\\h`, `\\t`
  - Extra whitespace normalization
- Non-CJK rows were dropped.

### en-zh_cn_vad_mix.csv
- Mixed dataset created for replay training:
  - 200k samples from `en-zh_cn_vad_clean.csv`
  - 200k samples from `en-zh_cn_vad_long_clean.csv`
  - Shuffled after sampling

## Training Summary

The final model (`vad-macbert-mix/best`) was obtained in three stages:

1. **Base training** on `en-zh_cn_vad_clean.csv`
2. **Long-text adaptation** on `en-zh_cn_vad_long_clean.csv`
3. **Replay mix** on `en-zh_cn_vad_mix.csv` (resume from stage 2)

### Final-stage Command (Replay Mix)

```
--model_name hfl/chinese-macbert-base
--output_dir train/vad-macbert-mix
--data_path train/en-zh_cn_vad_mix.csv
--epochs 4
--batch_size 32
--grad_accum_steps 4
--learning_rate 0.00001
--weight_decay 0.01
--warmup_ratio 0.1
--warmup_steps 0
--max_length 512
--eval_ratio 0.01
--eval_every 100
--eval_batches 200
--loss huber
--huber_delta 1.0
--shuffle_buffer 4096
--min_chars 2
--save_every 100
--log_every 1
--max_steps 5000
--seed 42
--dtype fp16
--num_rows 400000
--resume_from train/vad-macbert-long/best
--encoding utf-8
```

Training environment (conda `llm`):

- Python 3.10.19
- torch 2.9.1+cu130
- transformers 4.57.6

## Evaluation

Benchmark script: `train/vad_benchmark.py`

- Evaluation uses a fixed stride derived from `eval_ratio=0.01`
  (roughly 1 out of 100 samples).
- Length buckets by character count: 0–20, 20–40, 40–80, 80–120, 120–200,
  200–400, 400+

### Results (vad-macbert-mix/best)

**en-zh_cn_vad_clean.csv**

- mse_mean=0.043734
- mae_mean=0.149322
- pearson_mean=0.7335

**en-zh_cn_vad_long_clean.csv**

- mse_mean=0.031895
- mae_mean=0.131320
- pearson_mean=0.7565

Notes:
- `400+` bucket Pearson is unstable due to small sample size; interpret with care.

## Limitations

- Labels are derived from an English VAD teacher and transferred via parallel
  alignment, so they reflect the teacher’s bias and may not match human Chinese
  annotations.
- Subtitle corpora include translation artifacts and formatting noise; cleaned
  versions mitigate but do not fully remove this.
- Extreme-length sentences are under-represented; performance on 400+ chars
  is not reliable.

## Files in This Repo

- `config.json`
- `model.safetensors`
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`
- `training_args.json`
