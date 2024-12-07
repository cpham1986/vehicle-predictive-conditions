import torch
from torch import nn

class EngineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EngineLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)  # LSTM outputs
        out = self.fc(out[:, -1, :])  # Use the last hidden state for prediction
        return out
