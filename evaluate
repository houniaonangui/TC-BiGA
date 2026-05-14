import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


model.eval()

all_preds = []
all_labels = []


with torch.no_grad():

    for batch in test_loader:

        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        labels = batch['labels'].numpy()

        logits = model(
            input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(
            logits,
            dim=-1
        ).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)


acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='weighted')
rec = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')


print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")
