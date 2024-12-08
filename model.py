import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler

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

def predict(x, model):
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Scale inputs before feeding into the model
    x_scaled = scaler.fit_transform(x.reshape(-1, 1))  # Scale the input sequence
    
    # Convert scaled data to PyTorch tensor and reshape for LSTM input
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).view(1, -1, 1)  # (batch_size=1, sequence_length, input_size=1)
    
    # Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(x_tensor)  # Forward pass through the model
    
    # Convert outputs to NumPy array and inverse transform to original scale
    outputs_np = outputs.detach().numpy()
    outputs_rescaled = scaler.inverse_transform(outputs_np.reshape(-1, 1)).flatten()
    
    return outputs_rescaled

    