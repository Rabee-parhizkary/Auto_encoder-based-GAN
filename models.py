import torch
import torch.nn as nn
import ot
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Add a 1x1 convolution to match the dimensions if in_channels != out_channels
        if in_channels != out_channels:
            self.match_dimensions = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.match_dimensions = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # If needed, apply the 1x1 convolution to match dimensions
        if self.match_dimensions is not None:
            identity = self.match_dimensions(identity)

        out += identity
        out = self.relu(out)
        return out


# Encoder definition
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4096, 100)
        )

    def forward(self, x):
        return self.encoder(x)

# Decoder definition
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(100, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            ResBlock(256, 256),
            nn.Upsample(scale_factor=2),
            ResBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            ResBlock(128, 64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        return self.decoder(x)

# Autoencoder definition
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


#X is our distribution and Y is guasian
# Define the optimal mapping function
def optimal_mapping(latent_code_distribution, random_distribution, latent_code):
    L = 3  # Need L > 2 to allow the 2*y term, default is 1.4

    phi, G = ot.mapping.nearest_brenier_potential_fit(
        latent_code_distribution, random_distribution, X_classes=latent_code,
        strongly_convex_constant=0.6,
        gradient_lipschitz_constant=1.4,
        its=100, log=False, init_method='barycentric'
    )
    return phi, G




def generate_new_latent_code():
    print()


#def singular_value_clipping(brenier_potential):

