import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models import __models__
from datasets import __datasets__
from data.data_loader import create_data_loader
from omegaconf import OmegaConf
import wandb

# Load the configuration from config.yaml
config = OmegaConf.load("config.yaml")

# Initialize WandB with the project name from the configuration
wandb.init(project=config.wandb.project_name, config=config)

# Initialize distributed training if available
if torch.cuda.is_available():
    dist.init_process_group(backend="nccl")

# Define a function to train the model
def train_model(config):
    # Create the directory to save model checkpoints
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    # Initialize your model
    model = __models__[config.model.type]()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Wrap the model with DistributedDataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.MSELoss()  # Adjust the loss function as needed

    # Load your training dataset
    train_dataset = __datasets__[config.dataset.type](data_dir=config.dataset.data_dir, transform=None)
    train_loader = create_data_loader(train_dataset, batch_size=config.dataset.batch_size, shuffle=True)

    # Training loop
    for epoch in range(config.training.num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs, gts = batch['input'], batch['gt']
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Log training loss for this epoch
        average_loss = running_loss / len(train_loader)
        wandb.log({"train_loss": average_loss, "epoch": epoch + 1})

        # Save model checkpoint at specified intervals
        if (epoch + 1) % config.training.checkpoint_interval == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)  # Save the checkpoint to WandB

if __name__ == "__main__":
    train_model(config)
