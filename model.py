import torch
import torch.nn as nn
from transformers import AutoModel

from modules.fusion import FusionBlock


class ReviewerModel(nn.Module):

    def __init__(self, pretrained_model_name, num_labels=2):
        super(ReviewerModel, self).__init__()

        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name
        )

        self.hidden_size = self.encoder.config.hidden_size

        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.fusion = FusionBlock(self.hidden_size)

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask=None):

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask
        )

        base_features = outputs.last_hidden_state

        sequence_features, _ = self.gru(base_features)

        fused = self.fusion(
            base_features,
            sequence_features,
            attention_mask
        )

        pooled = fused.mean(dim=1)

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        return logits
