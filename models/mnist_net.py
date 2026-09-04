"""A small MNIST classifier: two Conv-BN-ReLU stages and a linear head.

Deliberately tiny (16.6K parameters) and returning raw logits, so
``nn.CrossEntropyLoss`` applies log-softmax itself. Its input is one 28x28
greyscale channel, which is what makes `dataset=mnist` the right thing to
train it on.
"""

import torch
import torch.nn as nn


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MnistNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(1, 16),
            nn.MaxPool2d(2),
            conv_bn_relu(16, 32),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
