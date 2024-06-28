import torch
import torch.nn as nn
from torchvision import models


class PneumoniaDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaDetectionModel, self).__init__()
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        for param in self.model.features.parameters():
            param.requires_grad = False

        n_inputs = self.model.classifier[6].in_features

        self.model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

