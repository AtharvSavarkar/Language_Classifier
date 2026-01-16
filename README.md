# 🇮🇳 Indic Language Classifier (23 Languages)

A high-throughput, production-grade language identification system for Indian languages, built by fine-tuning **IndicBERT v2**.

This repository provides:

- ✅ A **23-class language classifier** (English + 22 Indian languages)
- ✅ A **distributed, streaming inference pipeline** for very large JSONL datasets
- ✅ Supports **single-GPU and multi-GPU (DDP)** inference
- ✅ Outputs **language-wise sharded JSONL files**

## 🔤 Supported Languages (23)

```json
{
  "assamese": 0,
  "bengali": 1,
  "bodo": 2,
  "dogri": 3,
  "english": 4,
  "gujarati": 5,
  "hindi": 6,
  "kannada": 7,
  "kashmiri": 8,
  "kokani": 9,
  "maithili": 10,
  "malayalam": 11,
  "manipuri": 12,
  "marathi": 13,
  "nepali": 14,
  "oriya": 15,
  "punjabi": 16,
  "sanskrit": 17,
  "santali": 18,
  "sindhi": 19,
  "tamil": 20,
  "telugu": 21,
  "urdu": 22
}
```

## 🧠 Model Details

- Base Model: ai4bharat/IndicBERTv2-MLM-only
- Architecture: Transformer Encoder + Classification Head
- Task: Multiclass Language Identification (23-way)
- Input: Raw text
- Output: Language label + ID

## 📂 Input Format

The input directory should contain one or more .jsonl files.
Each line must be a JSON object containing a consistent text key (default: "text"):

```json
{"text": "यह एक उदाहरण वाक्य है"}
{"text": "This is an English sentence"}
{"text": "இது ஒரு தமிழ் வாக்கியம்"}
```

## 📤 Output Format

The script writes language-wise JSONL shards:

```text
output_dir/
 ├── hindi.rank0.jsonl
 ├── english.rank0.jsonl
 ├── tamil.rank1.jsonl
 ├── ...
```

Each output line contains the original JSON plus predictions:

```json
{
  "text": "यह एक उदाहरण वाक्य है",
  "predicted_id": 6,
  "predicted_label": "hindi"
}
```

## 🚀 Installation
```python
pip install torch transformers tqdm
```

## ▶️ Inference Usage
- Single GPU / CPU

```python
python classify.py \
  --model_path /path/to/model \
  --tokenizer_path /path/to/model \
  --input_dir /data/jsonl_inputs \
  --output_dir /data/lang_outputs \
  --label_map label_to_id.json \
  --batch_size 256
```

- Multi-GPU (DDP)

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 classify.py \
  --model_path /path/to/model \
  --tokenizer_path /path/to/model \
  --input_dir /data/jsonl_inputs \
  --output_dir /data/lang_outputs \
  --label_map label_to_id.json \
  --batch_size 256
```
Each GPU writes its own shard files which can be merged later.

| Argument           | Description             | Default  |
| ------------------ | ----------------------- | -------- |
| `--model_path`     | Path to model           | required |
| `--tokenizer_path` | Path to tokenizer       | required |
| `--input_dir`      | Folder with jsonl files | required |
| `--output_dir`     | Output folder           | required |
| `--label_map`      | JSON mapping label → id | required |
| `--batch_size`     | Batch size per GPU      | 256      |
| `--max_length`     | Max token length        | 512      |
| `--num_workers`    | DataLoader workers      | 64       |
| `--text_key`       | JSON key for text       | `"text"` |


