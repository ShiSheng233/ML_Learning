import torch.nn as nn


# Linear, ReLU
class COVIDModel(nn.Module):
    def __init__(self, input_size):
        super(COVIDModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)
