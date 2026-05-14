import torch
import torch.nn as nn


class FusionBlock(nn.Module):

    def __init__(self, hidden_size):
        super(FusionBlock, self).__init__()

        self.hidden_size = hidden_size

    def forward(self, base_features, sequence_features, attention_mask=None):

        raise NotImplementedError(
            "Fusion module is omitted in public reviewer version."
        )
