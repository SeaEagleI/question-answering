# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# Output layer: fc-layer
class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        """ FC layer as output_layer
        """
        super(FCLayer, self).__init__()
        self.qa_outputs = nn.Linear(in_features, out_features, bias=True)

    def forward(self, albert_outputs):
        logits = self.qa_outputs(albert_outputs)
        start_logits, end_logits = logits.split(1, dim=-1)
        # Return start & end logits
        return start_logits, end_logits
