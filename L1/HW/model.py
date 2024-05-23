import torch.nn as nn


class COVIDModel(nn.Module):
    def __init__(self, input_size):
        super(COVIDModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Sigmoid(),
            nn.Linear(16, 8),
            nn.Sigmoid(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)
