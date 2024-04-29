import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Define the neural network architecture
torch.set_default_dtype(torch.float64)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(6, 64)  # 6 input features, 64 hidden units
        self.fc2 = nn.Linear(64, 64)  # 64 hidden units, 64 hidden units
        self.fc3 = nn.Linear(64, 64)  # 64 hidden units, 64 hidden units
        self.fc4 = nn.Linear(64, 64)  # 64 hidden units, 64 hidden units
        self.fc5 = nn.Linear(64, 3)  # 64 hidden units, 3 output units

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation for the first layer
        x = torch.relu(self.fc2(x))  # ReLU activation for the second layer
        x = torch.relu(self.fc3(x))  # ReLU activation for the third layer
        x = torch.relu(self.fc4(x))  # ReLU activation for the fourth layer
        x = self.fc5(x)  # Final output layer, no activation
        return x


class SimpleNNWithLSTM(nn.Module):
    def __init__(self):
        super(SimpleNNWithLSTM, self).__init__()
        self.fc1 = nn.Linear(6, 10)  # 6 input features, 10 hidden units
        self.lstm = nn.LSTM(
            10, 10, batch_first=True
        )  # LSTM layer with input size 10 and hidden size 10
        self.fc2 = nn.Linear(10, 10)  # 10 hidden units, 3 output units
        self.fc3 = nn.Linear(10, 3)  # 10 hidden units, 3 output units

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation for the first layer
        x, _ = self.lstm(x.unsqueeze(0))  # LSTM layer, unsqueeze to add batch dimension
        x = torch.relu(x.squeeze(0))  # Remove batch dimension and apply ReLU activation
        x = torch.relu(self.fc2(x))  # ReLU activation for the second-to-last layer
        x = self.fc3(x)  # Final output layer, no activation
        return x


# Create an instance of the neural network
model = SimpleNN()
# model = SimpleNNWithLSTM() #TODO: not working!

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.000005)  # Adam optimizer

# Load the filtered state data
data = np.load("Project/filtered_state_EKF_CR3BP.npy")
np.random.shuffle(data.T)  # Shuffle each column randomly, before splitting
inputs = torch.tensor(data[:6, :]).t()
targets = torch.tensor(data[6:, :]).t()

# Lists to store training loss for plotting
train_loss_history = []

# Lists to store prediction errors for verification plot
prediction_errors = []

# Convert inputs and targets to PyTorch Dataset
dataset = TensorDataset(inputs, targets)

# Create DataLoader for batch processing
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 500000

for epoch in range(num_epochs):
    # Randomly sample a batch from the DataLoader
    batch_inputs, batch_targets = next(iter(dataloader))

    # Forward pass
    outputs = model(batch_inputs)
    loss = criterion(outputs, batch_targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate loss and prediction error for the entire dataset after each epoch
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_loss_history.append(loss.item())
        prediction_error = torch.abs(outputs - targets).mean().item()
        prediction_errors.append(prediction_error)

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.12f}, Prediction Error: {prediction_error:.12f}"
        )

# Optionally, save the trained model
torch.save(model.state_dict(), "Project/simple_nn_model.pth")

# Plot the training loss
plt.figure()
# plt.rc("text", usetex=True)
plt.semilogy(train_loss_history, color="blue")
plt.xlabel(r"Training Epoch  [-]")
plt.ylabel(r"Loss [-]")
plt.grid(True, which="both", linestyle="--")
# plt.savefig("Project/TrainingLoss.pdf", format="pdf")
plt.show()

# Plot the prediction errors for verification
plt.figure()
plt.semilogy(prediction_errors, color="red")
plt.xlabel(r"Training Epoch [-]")
plt.ylabel(r"Mean Prediction Error")
plt.grid(True, which="both", linestyle="--")
# plt.savefig("Project/PredictionError.pdf", format="pdf")
plt.show()
