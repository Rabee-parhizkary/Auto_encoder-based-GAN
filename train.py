import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import models
import matplotlib as  plt
from medmnist import RetinaMNIST
import ot
import numpy as np
# Assuming the Autoencoder code from previous response is already defined

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 50

# Data transforms for training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
dataset = RetinaMNIST(split="train", download=True,transform=transform, size=128)

# DataLoader is used to load the dataset
# for training
loader = torch.utils.data.DataLoader(dataset = dataset,
									batch_size = 32,
									shuffle = True)

# Initialize the autoencoder, loss function, and optimizer
autoencoder = models.Autoencoder()  # Move the model to GPU if available
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
#encoder = autoencoder.encoder()
# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        inputs, _ = data  # CIFAR-10 dataset returns image and label, but we only need the image
        # Move inputs to GPU if available


        # Zero the parameter gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = autoencoder(inputs)
        print(outputs.shape, inputs.shape)
        loss = criterion(outputs, inputs)  # MSE loss between input and reconstructed output
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the weights
        #latent = encoder(inputs)
        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0







    # Save example output images every epoch
    with torch.no_grad():
        sample_inputs = inputs[:8]
        reconstructed = autoencoder(sample_inputs)
        comparison = torch.cat([sample_inputs, reconstructed])
        vutils.save_image(comparison.cpu(), f'output/reconstructed_epoch_{epoch + 1}.png', nrow=8)

print('Finished Training')

# Save the trained model
torch.save(autoencoder.state_dict(), 'autoencoder.pth')


mu, sigma = 0, 0.1 # mean and standard deviation
#guasian = np.random.normal(mu, sigma, distribution.shape)

#brenier_potential = models.optimal_mapping(distribution,guasian, latent_code)
# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Using an Adam Optimizer with lr = 0.1
#optimizer = torch.optim.Adam(model.parameters(),
 #                            lr=1e-1,
  #                           weight_decay=1e-8)
