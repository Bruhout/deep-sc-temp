import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model fine-tuned on MNLI
model_name = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

premise = "A man is playing guitar on stage."
hypothesis = "A person is performing music."

# Tokenize the pair — RoBERTa uses </s></s> as the pair separator internally
inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)

with torch.no_grad():
    logits = model(**inputs).logits

probs = torch.softmax(logits, dim=-1)

# roberta-large-mnli label order: 0 = contradiction, 1 = neutral, 2 = entailment
labels = ["contradiction", "neutral", "entailment"]
pred_idx = probs.argmax(dim=-1).item()

print(f"Prediction: {labels[pred_idx]}")
print({labels[i]: round(probs[0][i].item(), 4) for i in range(len(labels))})