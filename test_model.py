import torch
from torch import nn
import matplotlib.pyplot as plt
from model import EngineLSTM
import pandas as pd 
from math import ceil
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# Hyperparameters
input_size = 80  # e.g., one feature (temperature, RPM, etc.)
hidden_size = 64
num_layers = 2
output_size = 20  # Single output for regression
batch_size = 32
num_epochs = 20
learning_rate = 0.001

# Data preparation
def prepare_time_series_data(data, input_seq_len, output_seq_len):
    X, y = [], []
    for i in range(len(data) - input_seq_len - output_seq_len):
        X.append(data[i:i+input_seq_len])  # Past 80 values
        y.append(data[i+input_seq_len:i+input_seq_len+output_seq_len])  # Next 20 values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

data = pd.read_csv("data/CSVLog_20241206_075200.csv",skiprows=1)
data = data[" Control module voltage (V)"].values
#data = data[" Engine RPM (RPM)"].values

# Remove any points that are 0
data = data[data != 0]

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Prepare the data
X, y = prepare_time_series_data(data, input_seq_len=input_size, output_seq_len=output_size)

dataset = torch.utils.data.TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, and optimizer
model = EngineLSTM(1, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:  # Fetch batches from train_loader
        batch_X = batch_X.unsqueeze(-1)  # Add feature dimension
        optimizer.zero_grad()
        outputs = model(batch_X)  # Forward pass
        loss = criterion(outputs, batch_y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


# Evaluation
model.eval()
test_loss = 0
predictions, actuals = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.unsqueeze(-1)  # Add feature dimension
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()
        predictions.extend(outputs.numpy())
        actuals.extend(batch_y.numpy())

print(f"Test Loss: {test_loss / len(test_loader):.4f}")

# Inverse scaling (if data is scaled)
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

# Reshape into arrays of size 20
predictions = predictions.reshape(-1, output_size)  # Reshape into (num_sequences, 20)
actuals = actuals.reshape(-1, output_size)          # Reshape into (num_sequences, 20)

plots = len(actuals)//output_size

# Create a single figure with 5 subplots (1 row, 5 columns)
fig, axes = plt.subplots(ceil(plots/5), 5, figsize=(5*plots//5, 5*3), sharey=True, dpi=500)
fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4)


for i in range(plots):
    # Plot on the ith subplot
    axes[i//5][i%5].plot(actuals[i], label="True Values", linestyle="--")
    axes[i//5][i%5].plot(predictions[i], label="Predicted Values")
    axes[i//5][i%5].set_title(f"True vs. Predicted Values {i+1}")
    axes[i//5][i%5].set_xlabel("Time Step")
    axes[i//5][i%5].grid(True)

#axes[0][0].legend()
