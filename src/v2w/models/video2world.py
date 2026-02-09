import torch.nn as nn
import torch.nn.functional as F


class Video2WorldModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Video2WorldModel, self).__init__()
        self.graph_encoder = nn.Linear(input_size, hidden_size)
        self.gcn_layer = nn.
    def forward(self, x):
        return x
    