# vad-macbert

基于 `hfl/chinese-macbert-base` 的中文 VAD（valence/arousal/dominance）回归模型。
输出 3 个连续值，目标对齐到教师模型 `RobroKools/vad-bert` 的 VAD 空间。

## 快速上手

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

## 模型信息

- 基座模型：`hfl/chinese-macbert-base`
- 任务：VAD 回归（3 维输出：valence, arousal, dominance）
- 头部：`AutoModelForSequenceClassification`，`num_labels=3`，`problem_type=regression`

## 数据来源与标注方式

### en-zh_cn_vad_clean.csv
- 来源：OpenSubtitles 英中平行语料。
- 标注：将英文句子输入 `RobroKools/vad-bert` 获取 VAD，再把该 VAD 赋给对应中文句子。

### en-zh_cn_vad_long.csv
- 由 `en-zh_cn_vad_clean.csv` 过滤长句得到（原始阈值未记录）。
- 根据长度统计推断最小长度为 32 字符，推测当时过滤条件为 `len >= 32`。

### en-zh_cn_vad_long_clean.csv
- 从 `en-zh_cn_vad_long.csv` 清洗得到，去掉字幕样式噪声：
  - ASS/SSA 标签块（如 `{\\fs..\\pos(..)}`，含不完整 `{`）
  - HTML 类标签（如 `<i>...</i>`）
  - 转义标记（`\\N`、`\\n`、`\\h`、`\\t`）
  - 多余空白归一化
- 非 CJK 内容已过滤。

### en-zh_cn_vad_mix.csv
- 回放混合数据：
  - `en-zh_cn_vad_clean.csv` 抽样 200k
  - `en-zh_cn_vad_long_clean.csv` 抽样 200k
  - 合并后再随机打乱

## 训练过程

最终模型 `vad-macbert-mix/best` 由三阶段训练获得：

1. **基础训练**：`en-zh_cn_vad_clean.csv`
2. **长句适配**：`en-zh_cn_vad_long_clean.csv`
3. **回放混合**：`en-zh_cn_vad_mix.csv`（从阶段 2 继续训练）

### 最终阶段命令（回放混合）

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

训练环境（conda `llm`）：

- Python 3.10.19
- torch 2.9.1+cu130
- transformers 4.57.6

## 评测

基准脚本：`train/vad_benchmark.py`

- 使用 `eval_ratio=0.01`（约 1/100 抽样）。
- 长度分桶（字符数）：0–20、20–40、40–80、80–120、120–200、200–400、400+

### 结果（vad-macbert-mix/best）

**en-zh_cn_vad_clean.csv**

- mse_mean=0.043734
- mae_mean=0.149322
- pearson_mean=0.7335

**en-zh_cn_vad_long_clean.csv**

- mse_mean=0.031895
- mae_mean=0.131320
- pearson_mean=0.7565

备注：
- `400+` 分桶样本量很少，Pearson 不稳定，仅供参考。

## 限制与注意事项

- VAD 标签来自英文教师模型并通过平行语料对齐，可能带有教师偏差，不等同于人工中文标注。
- 字幕语料存在翻译误差和格式噪声，清洗后仍可能残留。
- 超长句样本较少，`400+` 的表现不稳定。

## 目录文件

- `config.json`
- `model.safetensors`
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`
- `training_args.json`
