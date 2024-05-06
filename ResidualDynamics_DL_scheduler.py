import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# TODO: there's a relation with the EKF hyperparameters?

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
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, "min", factor=0.5, patience=10000, threshold=0.000001, min_lr=1e-8  # TODO: fix this!
# )

# Load the filtered state data
data = np.load("Project/filtered_state_EKF_CR3BP.npy")
np.random.shuffle(data.T)  # Shuffle each column randomly, before splitting
inputs = torch.tensor(data[:6, :]).t()
targets = torch.tensor(data[6:, :]).t()

# Lists to store training loss for plotting
train_loss_history = []

# Lists to store prediction errors for verification plot
prediction_errors = []

# Training loop
num_epochs = 200000
for epoch in range(num_epochs):
    # Forward pass
    optimizer.zero_grad()  # Clear gradients
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    # Store the loss for plotting
    train_loss_history.append(loss.item())

    # Calculate prediction errors for verification plot
    prediction_error = torch.abs(outputs - targets).mean().item()
    prediction_errors.append(prediction_error)

    # Update learning rate
    # scheduler.step(loss.item())

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
plt.savefig("Project/TrainingLoss.pdf", format="pdf")
plt.show()

# Plot the prediction errors for verification
plt.figure()
plt.semilogy(prediction_errors, color="red")
plt.xlabel(r"Training Epoch [-]")
plt.ylabel(r"Mean Prediction Error")
plt.grid(True, which="both", linestyle="--")
plt.savefig("Project/PredictionError.pdf", format="pdf")
plt.show()
