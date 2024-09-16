import wandb
import tqdm
import torch.optim as optim
import torch
from torchvision import datasets, transforms
import torch.nn as nn

sweep_config = {
    'method': 'random', #grid, random
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epoch': {
            'values': [5, 10]
        },
        'dropout': {
            'values': [0.3, 0.4, 0.5]
        },
        'conv_1': {
            'values': [16, 32, 64]
        },
        'conv_2': {
            'values': [16, 32, 64]
        },
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        'kernel_size': {
            'values': [(3, 3), (5, 5), (7, 7)]
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project="test_sweep")


class CNN_MNIST(nn.Module):
    def __init__(self, conv_1=32, conv_2=64, kernel_size=(3, 3), dropout=0.5):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, conv_1, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(conv_1, conv_2, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = None
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # Calculate the flattened size if fc1 is None
        if self.fc1 is None:
            # Get the shape after the conv and pool layers
            self.flattened_size = x.size(1) * x.size(2) * x.size(3)
            self.fc1 = nn.Linear(self.flattened_size, 128).to(x.device)  # Define fc1 with the calculated size
        x = x.view(-1,self.flattened_size)  # Flatten for FC layers
        x = self.dropout(torch.relu(self.fc1(x)))  # Apply dropout after the first FC layer
        x = self.fc2(x)
        return x


# Define the transform to convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the data
])

# Download and load the training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Download and load the test data
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# Optionally, you can create DataLoaders to iterate over the dataset in batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizers = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop,
}
# Training loop
# Initialize wandb with the sweep configuration
def in_train(config=None):
    # Initialize a new wandb run

    # Initialize the model with parameters from sweep config
    model = CNN_MNIST(conv_1=config.conv_1, conv_2=config.conv_2, kernel_size=config.kernel_size, dropout=config.dropout)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers[config.optimizer](model.parameters(), lr=0.001)

    # Load data (use previously defined train_loader and test_loader)
    for epoch in range(config.epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Track training statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Log training statistics to wandb
        wandb.log({"Train Loss": running_loss / len(train_loader), "Train Accuracy": 100. * correct / total})

        # Validate after each epoch
        validate(model, test_loader, criterion)

def validate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Log validation statistics to wandb
    wandb.log({"Validation Loss": running_loss / len(test_loader), "Validation Accuracy": 100. * correct / total})


def train():
    # Default values for hyperparameters we're going to sweep over
    config_defaults = {
        "conv_1": 32,
        "activation_1": "relu",
        "kernel_size": (3, 3),
        "pool_size": (2, 2),
        "dropout": 0.1,
        "conv_2": 64,
        "activation_out": "softmax",
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 6,
        "batch_size": 32
    }


    # Initialize a new wandb run
    wandb.init(config=config_defaults, group='first_sweeps')

    # config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    in_train(config)
wandb.agent(sweep_id, train)
