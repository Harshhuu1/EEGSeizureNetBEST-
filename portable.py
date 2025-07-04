import torch
import torch.nn as nn
from torchvision import models

# Custom model definition (same as in training)
class AdvancedClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.4):
        super(AdvancedClassifier, self).__init__()
        self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool
        feature_dim = self.backbone.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 16, feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        attention_weights = self.attention(features)
        features = features * attention_weights
        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Instantiate and load weights
model = AdvancedClassifier(num_classes=2, dropout_rate=0.4)  # Use actual num_classes
checkpoint = torch.load("best_advanced_model.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Convert to TorchScript
example_input = torch.randn(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("seizure_model_efficientnet.pt")

print("✅ TorchScript model saved as 'seizure_model_efficientnet.pt'")
