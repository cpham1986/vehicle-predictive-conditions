import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Generate or load sample data
data = {
    'time': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'miles_driven': np.cumsum(np.random.randint(30, 100, 100)),  # Simulated mileage
    'tire_pressure': np.random.uniform(30, 35, 100)  # Simulated pressure
}

df = pd.DataFrame(data)

# Convert timestamp to numeric feature (e.g., days since the start)
df['days_since_start'] = (df['time'] - df['time'].min()).dt.days

# Prepare features and target variable
X = df[['days_since_start', 'miles_driven']].values
y = df['tire_pressure'].values

# Scale the data for better neural network performance
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the Neural Network
class TirePressureNN(nn.Module):
    def __init__(self):
        super(TirePressureNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = TirePressureNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the model
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")

# Predict future values
future_days = np.array([101, 102, 103, 104, 105]).reshape(-1, 1)  # Days since start
future_miles = np.array([df['miles_driven'].iloc[-1] + i * 50 for i in range(1, 6)]).reshape(-1, 1)
future_data = np.hstack((future_days, future_miles))
future_data_scaled = scaler_X.transform(future_data)

future_data_tensor = torch.tensor(future_data_scaled, dtype=torch.float32)
with torch.no_grad():
    future_predictions_scaled = model(future_data_tensor)

# Inverse transform predictions
future_predictions = scaler_y.inverse_transform(future_predictions_scaled.numpy())

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(df['days_since_start'], df['tire_pressure'], label="Historical Data", color="blue")
plt.scatter(future_days, future_predictions, label="Future Predictions", color="red")
plt.xlabel("Days Since Start")
plt.ylabel("Tire Pressure (PSI)")
plt.title("Tire Pressure Prediction")
plt.legend()
plt.grid()
plt.show()
