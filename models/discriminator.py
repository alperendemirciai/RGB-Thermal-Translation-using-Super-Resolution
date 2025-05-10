# discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, target_channels=4, base_filters=64):
        super().__init__()
        channels = in_channels + target_channels  # Concatenate input and target/generated image

        def discriminator_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, base_filters, normalize=False),
            *discriminator_block(base_filters, base_filters * 2),
            *discriminator_block(base_filters * 2, base_filters * 4),
            *discriminator_block(base_filters * 4, base_filters * 8),
            nn.Conv2d(base_filters * 8, 1, kernel_size=4, padding=1)  # Output 1-channel "realness" map
        )

    def forward(self, x, y):
        # Concatenate input (e.g. RGB) and target/generated (e.g. IR) along channel dimension
        x = torch.cat([x, y], dim=1)
        return self.model(x)

if __name__ == "__main__":
    # Example usage
    discriminator = Discriminator()
    rgb_image = torch.randn(1, 3, 256, 256)  # Batch of RGB images
    ir_image = torch.randn(1, 4, 256, 256)   # Batch of IR images
    output = discriminator(rgb_image, ir_image)
    print(output.shape)  # Should be (1, 1, 16, 16) for a 256x256 input image