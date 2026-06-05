import torch
import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self, num_classes, input_channel=3, output_channel=32, dropout=0):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(output_channel, output_channel * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(output_channel * 2, output_channel * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(output_channel * 4, output_channel * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel * 8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(output_channel * 8, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
