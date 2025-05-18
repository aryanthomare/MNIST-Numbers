from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModel(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # TODO: define each layer

        #28 x 28
        self.conv1 =  nn.Conv2d(
            in_channels=1,       # 1 Input channels
            out_channels=6,     # 16 filters
            kernel_size=5,       # 5x5 kernel
            stride=1,            # Stride of 2
            padding=2            # Padding of 2 to preserve spatial dimensions
        )
        #16 x 28 x 28

        self.pool = nn.MaxPool2d(
            kernel_size=2,        # 2x2 pooling
            stride=2             # Stride of 2
        )
        #16 x 14 x 14
        self.conv2 = nn.Conv2d(
            in_channels=6,       # 16 Input channels
            out_channels=36,     # 64 filters
            kernel_size=7,       # 5x5 kernel
            stride=1,            # Stride of 2
            padding=1            # Padding of 2 to preserve spatial diensions
        )
        #36 x 14 x 14
        
        self.conv3 = nn.Conv2d(
            in_channels=36,       # 64 Input channels
            out_channels=12,     # 64 filters
            kernel_size=3,       # 5x5 kernel
            stride=2,            # Stride of 2
            padding=2            # Padding of 2 to preserve spatial diensions
        )

        self.fc_1 = nn.Linear(192, 80)
        self.fc_2 = nn.Linear(80, 10)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.normal_(conv.weight, mean=0.0, std=1/sqrt(conv.kernel_size[0] * conv.kernel_size[1] * conv.in_channels))
            nn.init.constant_(conv.bias, 0.0)

        # TODO: initialize the parameters for [self.fc_1]
        nn.init.normal_(self.fc_1.weight, mean=0.0, std=1/sqrt(self.fc_1.in_features))
        nn.init.normal_(self.fc_2.weight, mean=0.0, std=1/sqrt(self.fc_2.in_features))
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.constant_(self.fc_2.bias, 0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N, C, H, W = x.shape


        # TODO: forward pass
        c1 = F.relu(self.conv1(x))

        p1 = self.pool(c1)
        c2 = F.relu(self.conv2(p1))
        p2 = self.pool(c2)
        c3 = F.relu(self.conv3(p2))
        p3 = self.pool(c3)
        flatten = torch.flatten(c3, start_dim=1)
        fc = self.fc_1(flatten)
        fc = F.relu(fc)
        fc = self.fc_2(fc)

        if False:
            print("Input shape: ",x.shape)
            print("C1 shape: ",c1.shape)
            print("P1 shape: ",p1.shape)
            print("C2 shape: ",c2.shape)
            print("P2 shape: ",p2.shape)
            print("C3 shape: ",c3.shape)
            print("P3 shape: ",p3.shape)
            print("Flatten shape: ",flatten.shape)
            print("FC1 shape: ",fc.shape)
        return fc