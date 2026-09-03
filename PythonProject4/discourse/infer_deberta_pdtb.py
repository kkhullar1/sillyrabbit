from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_DIR
    / "discourse"
    / "models"
    / "original_deberta_pdtb"
    / "checkpoint-1590"
)

TOKENIZER_NAME = "microsoft/deberta-v3-large"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.eval()


def predict_relation_distribution(arg1, arg2):
    inputs = tokenizer(
        str(arg1),
        str(arg2),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    distribution = {}

    for idx, probability in enumerate(probabilities):
        label = model.config.id2label[idx]
        distribution[label] = float(probability.item())

    return distribution


def predict_relation(arg1, arg2):
    distribution = predict_relation_distribution(arg1, arg2)
    return max(distribution, key=distribution.get)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python discourse/infer_deberta_pdtb.py ARG1 ARG2"
        )

    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    distribution = predict_relation_distribution(arg1, arg2)
    prediction = max(distribution, key=distribution.get)

    print(prediction)

    for label, probability in sorted(distribution.items()):
        print(f"{label}\t{probability:.6f}")
