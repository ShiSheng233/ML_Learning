import torch.nn as nn


# Linear, Sigmoid
class COVIDModel(nn.Module):
    def __init__(self, input_size):
        super(COVIDModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Sigmoid(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)
