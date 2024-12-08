import torch
from torch import nn
import matplotlib.pyplot as plt
from model import EngineLSTM
import pandas as pd 
from math import ceil
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
from app import load_config

# Load the configuration
config = load_config()
input_size = config['input_sequence_length']
hidden_size = config['hidden_size']
num_layers = config['num_layers']
output_size = config['output_sequence_length']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
learning_rate = config['learning_rate']

EVAL = 1

# Data preparation
def prepare_time_series_data(data, input_seq_len, output_seq_len):
    X, y = [], []
    for i in range(len(data) - input_seq_len - output_seq_len):
        X.append(data[i:i+input_seq_len])  # Past 80 values
        y.append(data[i+input_seq_len:i+input_seq_len+output_seq_len])  # Next 20 values
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

def make_train_model(model_data_in, model_name):
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    model_data = scaler.fit_transform(model_data_in.reshape(-1, 1)).flatten()

    # Prepare the data
    X, y = prepare_time_series_data(model_data, input_seq_len=input_size, output_seq_len=output_size)

    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model and optimizer
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
        if(epoch % 5 == 4): print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.6f}")

    if(EVAL == 1):
        eval_model(model, model_name, test_loader, scaler, criterion)

    torch.save(model.state_dict(), f"models/{model_name}_model.pth")

def eval_model(model, model_name, test_loader, scaler, criterion):
    model.eval()
    test_loss = 0
    predictions, actuals = [], []
    x = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.unsqueeze(-1)  # Add feature dimension
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            x.extend(batch_X.squeeze(1))
            predictions.extend(outputs.numpy())
            actuals.extend(batch_y.numpy())

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

    # Inverse scaling
    x = scaler.inverse_transform(np.array(x).reshape(-1, 1)).flatten()
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    # Reshape into arrays of size 20
    x = x.reshape(-1, input_size)
    predictions = predictions.reshape(-1, output_size)
    actuals = actuals.reshape(-1, output_size)

    # Randomly select 25 indices
    total_sequences = predictions.shape[0]
    num_plots = min(25, total_sequences)  # Ensure not to exceed the total number of sequences
    selected_indices = np.random.choice(total_sequences, num_plots, replace=False)
    selected_indices.sort()

    # Determine grid size dynamically
    rows = ceil(num_plots / 5)
    cols = min(num_plots, 5)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), dpi=100)

    fig.suptitle(f"{model_name}: True vs. Predicted Values Across Randomly Selected Sequences", fontsize=16, fontweight='bold')

    # Ensure `axes` is always iterable (even if 1 row or 1 column)
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    for plot_idx, seq_idx in enumerate(selected_indices):
        row, col = divmod(plot_idx, 5)
        ax = axes[row][col]  # Get the correct subplot
        ax.plot(range(0, input_size), x[seq_idx], label="Input Values")
        ax.plot(range(input_size, input_size + output_size), actuals[seq_idx], label="True Values", linestyle="--")
        ax.plot(range(input_size, input_size + output_size), predictions[seq_idx], label="Predicted Values")
        ax.set_title(f"Sequence {seq_idx+1}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust spacing to fit the title
    plt.show()  # Ensure the plots are displayed


if 0 and len(sys.argv) == 1:
    print(f"Usage: python3 {sys.argv[0]} <data_file_name>.csv")
    sys.exit()
#csv_file = sys.argv[1]

csv_file = "data/CSVLog_20241207_131020.csv"
data = pd.read_csv(csv_file,skiprows=1)

for col in data.columns[1:]:
    name = col.split('(')[0].strip().replace(' ','_')
    print(name)
    #print(data[col])
    make_train_model(data[col].values, name)


plt.show()
