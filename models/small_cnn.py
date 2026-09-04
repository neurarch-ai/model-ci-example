"""A small CIFAR-10 CNN with BatchNorm.

Each stage is Conv -> BatchNorm -> ReLU, with the conv bias off because
BatchNorm carries its own shift term. The head returns class probabilities.
"""

import torch
import torch.nn as nn


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(3, 32),
            conv_bn_relu(32, 32),
            nn.MaxPool2d(2),
            conv_bn_relu(32, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(2),
            conv_bn_relu(256, 192),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.CrossEntropyLoss()(logits, targets)


if __name__ == "__main__":
    model = SmallCNN()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(out.shape, loss(out, torch.randint(0, 10, (4,))).item())
