import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import pywt
import matplotlib.pyplot as plt

class VGG16Morlet(nn.Module):
    def __init__(self):
        # Initialize the model with pretrained ImageNet Weights
        super(VGG16Morlet, self).__init__()
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 3)
        self.model = model

        # Pre load colormap and scales
        self.cmap = plt.get_cmap('nipy_spectral')
        self.scales = np.arange(1, 24)

    def _generate_scalogram_tensor(self, signal_tensor):
        # Convert tensor to numpy (CPU)
        signal = signal_tensor.cpu().numpy()

        # Perform CWT using pywt and morlet wavelet
        coeffs, _ = pywt.cwt(signal, self.scales, 'morl', method='conv')

        # Normalization
        max_val = np.abs(coeffs).max()
        if max_val > 0:
            coeffs_norm = 0.5 + 0.5 * (coeffs / max_val)
        else:
            coeffs_norm = coeffs

        # Apply colormap
        img_rgba = self.cmap(coeffs_norm)
        img_rgb = img_rgba[:, :, :3]

        # Convert back to tensor
        img_tensor = torch.from_numpy(img_rgb).float()
        img_tensor = img_tensor.permute(2, 0, 1)

        return img_tensor

    def forward(self, x):
        """
        x: Input EEG signals of shape (Batch_Size, 400) or (Batch_Size, 1, 400)
        """
        # Ensure input is 2D (Batch, Time) for processing
        if x.dim() == 3:
            x = x.squeeze(1)

        # List to hold the processed images
        batch_images = []

        # Process each signal in the batch
        with torch.no_grad():
            for i in range(x.size(0)):
                # Generate scalogram for this specific signal
                img = self.generate_scalogram_tensor(x[i])
                batch_images.append(img)

            # Stack into a single batch tensor: (Batch, 3, 23, 400)
            batch_tensor = torch.stack(batch_images)

            # Move back to the same device as the model
            batch_tensor = batch_tensor.to(x.device)

        # Resize to 224x224 (Required by VGG16)
        batch_tensor = F.interpolate(batch_tensor, size=(224, 224), mode='bilinear', align_corners=False)

        # Pass through VGG model
        return self.model(batch_tensor)
