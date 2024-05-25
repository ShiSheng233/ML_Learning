# HW1

## Description

I get utility functions `util.py` from Hung-yi Lee's sample code.

## Model

```python
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
```

I just use a simple linear model.

However, I also tried polynomial regression, but I don't know how to construct the model.

## Data Selection

```python
feat_column = [i for i in range(raw_x_train.shape[1]) if i == 0 or i >= 35]  # ignore state features
```
I ignore the state location features.

## Problems

- How to find the best activation function for linear layer
- How to configure optimizer
- How to set hyperparameters (especially learning rate, batch size)
- How can I make use of the data column 1-34(location), or can I put weight on the location parameter
- How to construct polynomial regression model

## Findings

- `model.eval()` is used to kill dropout and batch normalization, not necessary for my simple model.
