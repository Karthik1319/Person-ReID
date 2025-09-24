import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(FeatureExtractor, self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet50 = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim: int = 2048):
        super(PolicyNetwork, self).__init__()
        self.fc_mean = nn.Linear(feature_dim, feature_dim)
        self.fc_std = nn.Linear(feature_dim, feature_dim)

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        mean = self.fc_mean(x)
        std = F.softplus(self.fc_std(x)) + 1e-5 # Add epsilon for stability
        return mean, std

class ReIDModel(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor, num_classes: int):
        """
        Args:
            feature_extractor (FeatureExtractor): The backbone model.
            num_classes (int): The number of unique person IDs in the training set.
        """
        super(ReIDModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # --- NEW ---
        # Add a classifier layer for supervised pre-training.
        # The feature dimension of ResNet-50 is 2048.
        self.classifier = nn.Linear(2048, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method is used during evaluation and RL training.
        It returns only the final feature vector.
        """
        feature_map = self.feature_extractor(x)
        return self.gap(feature_map).flatten(1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for supervised training.
        It returns both the feature vector and the classification scores.
        """
        feature_map = self.feature_extractor(x)
        features = self.gap(feature_map).flatten(1)
        # Get the classification scores (logits)
        class_scores = self.classifier(features)
        return features, class_scores
