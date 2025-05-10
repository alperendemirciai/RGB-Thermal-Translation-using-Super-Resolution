import torch
import torch.nn as nn
from UNet import UNet

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, base_filters=8, upsampling_method="pixelshuffle"):
        super().__init__()
        self.model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
            upsampling_method=upsampling_method
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Example usage
    generator = Generator()
    rgb_image = torch.randn(1, 3, 256, 256)  # Batch of RGB images
    output = generator(rgb_image)
    print(output.shape)  # Should be (1, 4, 256, 256) for a 256x256 input image