import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG16(nn.Module):
    def __init__(self):
        # Initialize the model with pretrained ImageNet Weights
        super(VGG16, self).__init__()
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 3)
        self.model = model

    def forward(self, input):
        return self.input(model)

