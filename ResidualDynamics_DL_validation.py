import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network architecture
torch.set_default_dtype(torch.float64)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(6, 10)  # 6 input features, 10 hidden units
        self.fc2 = nn.Linear(10, 10)  # 10 hidden units, 10 hidden units
        self.fc3 = nn.Linear(10, 10)  # 10 hidden units, 10 hidden units
        self.fc4 = nn.Linear(10, 10)  # 10 hidden units, 10 hidden units
        self.fc5 = nn.Linear(10, 3)  # 10 hidden units, 3 output units

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation for the first layer
        x = torch.relu(self.fc2(x))  # ReLU activation for the second layer
        x = torch.relu(self.fc3(x))  # ReLU activation for the third layer
        x = torch.relu(self.fc4(x))  # ReLU activation for the fourth layer
        x = self.fc5(x)  # Final output layer, no activation
        return x


# Create an instance of the neural network
model = SimpleNN()

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer

# Load the filtered state data
data = np.load("Project/filtered_state_EKF_CR3BP.npy")
np.random.shuffle(data.T)  # Shuffle each column randomly, before splitting

# Define the sizes for training and validation sets
train_size = int(0.9 * len(data.T))
val_size = len(data.T) - train_size

# Split the data into inputs and targets
train_inputs = torch.tensor(data[:6, :train_size]).t()
train_targets = torch.tensor(data[6:, :train_size]).t()
val_inputs = torch.tensor(data[:6, train_size:]).t()
val_targets = torch.tensor(data[6:, train_size:]).t()

# Lists to store training loss for plotting
train_loss_history = []

# Lists to store prediction errors for verification plot
prediction_errors = []

# Training loop
num_epochs = 300000
for epoch in range(num_epochs):
    # Forward pass
    model.train()
    optimizer.zero_grad()  # Clear gradients
    outputs = model(train_inputs)
    loss = criterion(outputs, train_targets)

    # Backward pass and optimization
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    # Store the loss for plotting
    train_loss_history.append(loss.item())

    # Calculate prediction error for validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_inputs)
        prediction_error = torch.abs(val_outputs - val_targets).mean().item()
        prediction_errors.append(prediction_error)

    # Logging every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.14f}, Prediction Error: {prediction_error:.14f}"
        )

# Optionally, save the trained model
torch.save(model.state_dict(), "Project/simple_nn_model.pth")

# Plot the training loss
plt.figure()
# plt.rc("text", usetex=True)
plt.semilogy(train_loss_history, color="blue")
plt.xlabel(r"Training Epoch")
plt.ylabel(r"Loss, Training Dataset")
plt.grid(True, which="both", linestyle="--")
plt.savefig("Project/TrainingLoss.pdf", format="pdf")
plt.show()

# Plot the prediction errors for verification
plt.figure()
plt.semilogy(prediction_errors, color="red")
plt.xlabel(r"Training Epoch")
plt.ylabel(r"Mean Prediction Error, Validation Dataset")
plt.grid(True, which="both", linestyle="--")
plt.savefig("Project/PredictionError.pdf", format="pdf")
plt.show()
