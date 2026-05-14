import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)

from sklearn.metrics import accuracy_score, f1_score

from model import ReviewerModel


MODEL_NAME = "cardiffnlp/twitter-roberta-base-2022-154m"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model = ReviewerModel(
    MODEL_NAME,
    num_labels=2
).to(DEVICE)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.01
)


loss_fn = torch.nn.CrossEntropyLoss(
    label_smoothing=0.1
)


EPOCHS = 5


scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)


scaler = torch.amp.GradScaler('cuda')


for epoch in range(EPOCHS):

    model.train()

    for batch in train_loader:

        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):

            logits = model(
                input_ids,
                attention_mask=attention_mask
            )

            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)

        scaler.update()

        scheduler.step()

    print(f"Epoch {epoch+1} finished.")
