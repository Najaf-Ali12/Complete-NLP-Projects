---
library_name: transformers
license: apache-2.0
base_model: distilbert-base-uncased
tags:
- text-classification
- customer-support
- distilbert
- generated_from_trainer
metrics:
- accuracy
- precision
- recall
- f1
model-index:
- name: QueryCategorizer
  results: []
datasets:
- bitext/Bitext-customer-support-llm-chatbot-training-dataset
language:
- en
pipeline_tag: text-classification
---

# QueryCategorizer

**QueryCategorizer** is a fine-tuned [DistilBERT](https://huggingface.co/distilbert-base-uncased) model that classifies customer support queries into one of **9 departments**. It is designed to help auto-route incoming support tickets to the correct team.

- **Base model:** `distilbert-base-uncased`
- **Task:** Multi-class text classification (9 classes)
- **Language:** English
- **Dataset:** [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- **License:** Apache 2.0

## Labels

The model predicts one of the following departments:

| Label | Example query |
|-------|---------------|
| ACCOUNT | "I can't log in to my account" |
| CONTACT | "How can I speak to a human agent?" |
| DELIVERY | "When will my package arrive?" |
| FEEDBACK | "I want to leave a review" |
| INVOICE | "Where can I download my invoice?" |
| ORDER | "I want to change my order" |
| PAYMENT | "My payment failed" |
| REFUND | "I want my money back" |
| SHIPPING | "Can I change my shipping address?" |

> The example queries are illustrative and written for this README.

## Quick Start

### Using the pipeline

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="NajafAli01/QueryCategorizer")

result = classifier("I want to get a refund for my last order")
print(result)
# [{'label': 'REFUND', 'score': 0.99}]
```

### Loading the model directly

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "NajafAli01/QueryCategorizer"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

text = "Where is my package?"
inputs = tokenizer(text, return_tensors="pt", truncation=True)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_id = logits.argmax(dim=-1).item()
print(model.config.id2label[predicted_id])
```

Install the requirements with:

```bash
pip install transformers torch
```

## Intended Uses

- Routing incoming customer support emails and messages to the right department
- Prototyping multi-class text classification pipelines
- Educational demonstrations of fine-tuning transformer models

## Limitations

- The model only **classifies** a message. It does not forward it to a department.
- The dataset has 11 categories, but only **9** were used. **CANCEL** and **SUBSCRIPTION** were removed because they had too few samples.
- Queries about cancellations or subscriptions will therefore be assigned to one of the 9 available labels, which may be wrong.
- English only.
- The dataset is largely templated and synthetic, so the very high scores below may not carry over to real, messy customer messages. Test the model on your own data before relying on it in production.

## Training Data

The Bitext customer support dataset was split as follows:

| Split | Share |
|-------|-------|
| Train | 80% |
| Validation | 10% |
| Test | 10% |

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-05 |
| Train batch size | 16 |
| Eval batch size | 16 |
| Seed | 42 |
| Optimizer | AdamW (fused), betas=(0.9, 0.999), epsilon=1e-08 |
| LR scheduler | Linear, 500 warmup steps |
| Epochs | 10 (max) |

## Results

Best results on the evaluation set:

| Metric | Score |
|--------|-------|
| Loss | 0.0038 |
| Accuracy | 0.9996 |
| Precision | 0.9996 |
| Recall | 0.9996 |
| F1 | 0.9996 |

### Training log

| Training Loss | Epoch | Step | Validation Loss | Accuracy | Precision | Recall | F1 |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:---------:|:------:|:------:|
| 0.0079 | 1.0 | 1125 | 0.0053 | 0.9988 | 0.9988 | 0.9988 | 0.9988 |
| 0.0036 | 2.0 | 2250 | 0.0061 | 0.9984 | 0.9984 | 0.9984 | 0.9984 |
| 0.0009 | 3.0 | 3375 | 0.0038 | 0.9996 | 0.9996 | 0.9996 | 0.9996 |
| 0.0005 | 4.0 | 4500 | 0.0038 | 0.9992 | 0.9992 | 0.9992 | 0.9992 |
| 0.0001 | 5.0 | 5625 | 0.0029 | 0.9996 | 0.9996 | 0.9996 | 0.9996 |
| 0.0000 | 6.0 | 6750 | 0.0028 | 0.9996 | 0.9996 | 0.9996 | 0.9996 |

## Framework Versions

- Transformers 5.16.1
- PyTorch 2.11.0+cu128
- Datasets 4.8.5
- Tokenizers 0.23.1

## Author

**Najaf Ali**
Hugging Face: [NajafAli01](https://huggingface.co/NajafAli01)
GitHub: [Najaf-Ali12](https://github.com/Najaf-Ali12)
