"""Five-layer AOD-Net, following Li et al. (ICCV 2017)."""
import torch
from torch import nn

class AODNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv2d(6, 3, 5, padding=2)
        self.conv4 = nn.Conv2d(6, 3, 7, padding=3)
        self.conv5 = nn.Conv2d(12, 3, 3, padding=1)

    def forward(self, x):
        a = self.relu(self.conv1(x))
        b = self.relu(self.conv2(a))
        c = self.relu(self.conv3(torch.cat((a, b), dim=1)))
        d = self.relu(self.conv4(torch.cat((b, c), dim=1)))
        k = self.relu(self.conv5(torch.cat((a, b, c, d), dim=1)))
        return self.relu(k * x - k + 1)

    def initialize_identity(self):
        # A transparent initial baseline: K(x)=1 makes output equal input.
        nn.init.zeros_(self.conv5.weight)
        nn.init.ones_(self.conv5.bias)
