# Details of my finetuning a sentiment model using trainer library of hugging face
# Model Files

This folder contains the fine-tuned NLP model used in the **[Finetuning Sentiment analysis model]** project of the [Complete-NLP-Projects](https://github.com/Najaf-Ali12/Complete-NLP-Projects) repository.

- **Task:** Sentiment Analysis (3-class text classification: Negative, Neutral, Positive)
- **Model architecture:** DistilBERT (`DistilBertForSequenceClassification`), 6 layers, 12 attention heads, hidden size 768, vocab size 30,522, max 512 tokens
- **Framework:** Hugging Face Transformers (PyTorch)
- **Dataset:** [dataset name / link]

---

## ⚠️ Important: `model.safetensors` is not included in this repository

The main weights file, **`model.safetensors`**, is too large for GitHub's file size limit (100 MB per file), so it could **not** be uploaded here. All other files (config, tokenizer, etc.) are included.

Without `model.safetensors` the model **will not load**. Download it from the link below and place it in this folder.

### 📥 Download

**[⬇️ Download model.safetensors]**
(https://huggingface.co/NajafAli01/my-sentiment-model/tree/main)

File size: ~268 MB

### Setup

After downloading, your folder should look like this:

```
model/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.txt
├── model.safetensors   <-- download and add this file
└── README.md
```

> File names may differ slightly depending on the tokenizer used. Keep everything that is in this folder and just add `model.safetensors` next to it.

---

## 📦 Folder Contents

| File | Description |
|------|-------------|
| `config.json` | Model architecture and configuration (DistilBERT, 3 labels) |
| `tokenizer.json` / `vocab.txt` | Tokenizer vocabulary and settings |
| `tokenizer_config.json` | Tokenizer configuration |
| `special_tokens_map.json` | Special tokens (`[CLS]`, `[SEP]`, `[PAD]`, etc.) |
| `model.safetensors` | Trained model weights (**download separately**) |

---

## 🚀 Usage

Install the requirements:

```bash
pip install transformers torch safetensors
# config was saved with transformers 5.x, so use a recent version
```

Load the model from this folder:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./model"  # path to this folder

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

text = "Your input text here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)

prediction = torch.argmax(outputs.logits, dim=1).item()
print("Predicted class:", prediction)
```

Or use the pipeline API:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="./model", tokenizer="./model")
print(classifier("Your input text here"))
```

The model predicts one of three labels:

| ID | Label |
|----|-------|
| 0 | NEGATIVE |
| 1 | NEUTRAL |
| 2 | POSITIVE |

To print the label name instead of the ID:

```python
print("Sentiment:", model.config.id2label[prediction])
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.88% |
| Precision |0.89% |
| Recall | 0.88% |
| F1-score |0.88% |

---

## 🛠️ Training Details

- **Epochs:** 3
- **Batch size:** 16
- **Max sequence length:** the model supports up to 512
- **Hardware:**  Google Colab T4 GPU

---

## ❓ Troubleshooting

- **`OSError: Error no file named pytorch_model.bin, model.safetensors ...`**
  `model.safetensors` is missing. Download it and place it in this folder.
- **File seems corrupted or fails to load:** Re-download it and make sure the download fully completed. Check the file size matches ~268 MB.
- **Path errors:** Make sure `model_path` points to the folder that contains `config.json` and `model.safetensors`.

---

## 👤 Author

**Najaf Ali**
GitHub: [@Najaf-Ali12](https://github.com/Najaf-Ali12)

