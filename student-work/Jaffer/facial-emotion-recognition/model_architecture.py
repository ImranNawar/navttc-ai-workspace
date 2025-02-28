import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CLIPClassifier, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # input size of the linear layer based on expected output dimensions
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=x)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        out = self.fc(image_features)
        return out