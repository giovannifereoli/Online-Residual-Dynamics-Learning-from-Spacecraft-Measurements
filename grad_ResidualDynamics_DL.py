import torch
import torch.nn as nn
from torch.autograd import grad


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


# Load the model
model = SimpleNN()
model.load_state_dict(torch.load("Project/simple_nn_model.pth"))
model.eval()

# Count the number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters:", total_params)

# Estimate memory usage (assuming 4 bytes per parameter)
memory_usage_bytes = total_params * 4
print("Estimated memory usage:", memory_usage_bytes, "bytes")

# Define your specific input
input_data = torch.tensor(
    [1.0, 0.0, 0.0, 0.0, 0.5, 0.0], requires_grad=True
)  # Example input

# Forward pass
output = model(input_data)

# Compute the Jacobian matrix
jacobian = torch.zeros(output.size(0), input_data.size(0))
for i in range(output.size(0)):
    gradients = grad(output[i], input_data, create_graph=True)[0]
    jacobian[i, :] = gradients

print("Output:", output.detach().numpy())
print("Jacobian Matrix:")
print(jacobian.detach().numpy())

# TODO: create a EKF-NN model
