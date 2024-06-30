# HW2: Classification

## Model

The first thing I thought of was using LSTM to solve the classification problem. And the model is defined as follows:

```python
# data
concat_nframes = 11
train_ratio = 0.8

# model
input_dim = 39 * concat_nframes
hidden_layers = 2  # LSTM layers
hidden_dim = 64

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=4, hidden_dim=256, batch_size = 8):
        super(LSTMClassifier, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        
        # input
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # lstm
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, hidden_layers, batch_first=True)
        
        # output
        self.bc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )


    def forward(self, x):
        x = self.fc(x)
        out, _ = self.lstm(x)
        out = self.bc(out)
        return out
```
However, the accuracy is quite bad (score around 0.55 after 10 epoch training), after looking for some information, I found that I didn't implement Dropout layer and layer normalization. 

This could result in overfitting and the model is not able to generalize well.

In version 2, I turn LSTM to BiLSTM, added layer normalization and better full connect layer.

After doing some research, I found that my model structure is not good, so I reorganize the model, added more hidden dim for more parameters.

With more contacted frame of input data, the trained model is able to achieve higher score.

## Findings

## Improvements

- Try auto adjust the learning rate (torch.optim.lr_scheduler)
