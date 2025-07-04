import os
import random
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------- Focal Loss -----------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ------------------- EfficientNet-B0 Model -------------------
class FastClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feat_dim = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base(x)


# --------------------- Set Seeds ---------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------ MAIN ------------------------
def main():
    set_seed()

    # 🛠 Config
    DATA_DIR = r"C:\\Users\\ASUS\\PycharmProjects\\Langchainmodels\\chb01_fast_download\\dataset"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Using device: {DEVICE}")

    # 📦 Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 📊 Dataset + Stratified Split
    full_dataset = ImageFolder(DATA_DIR, transform=train_transform)
    val_dataset_full = ImageFolder(DATA_DIR, transform=val_transform)
    all_labels = [label for _, label in full_dataset]
    class_counts = Counter(all_labels)
    num_classes = len(class_counts)

    # 📚 Stratified split
    indices_per_class = {cls: [] for cls in range(num_classes)}
    for idx, (_, label) in enumerate(full_dataset):
        indices_per_class[label].append(idx)

    train_idx, val_idx = [], []
    for cls, indices in indices_per_class.items():
        np.random.shuffle(indices)
        split = int(0.85 * len(indices))
        train_idx.extend(indices[:split])
        val_idx.extend(indices[split:])

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(val_dataset_full, val_idx)

    # ⚖️ Weighted sampler
    train_labels = [all_labels[i] for i in train_idx]
    beta = 0.9999
    eff_num = 1.0 - np.power(beta, list(Counter(train_labels).values()))
    weights_per_class = (1.0 - beta) / np.array(eff_num)
    weights = [weights_per_class[full_dataset[i][1]] for i in train_idx]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # 🧠 Model
    model = FastClassifier(num_classes).to(DEVICE)

    # ⚙️ Loss, Optimizer, Scheduler
    class_weights = torch.FloatTensor([
        len(train_labels) / (num_classes * class_counts[i]) for i in range(num_classes)
    ]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.003, steps_per_epoch=len(train_loader), epochs=25)

    # 🔁 Training Loop
    best_val_acc, patience, trigger = 0, 5, 0
    for epoch in range(1, 26):
        model.train()
        total, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        train_acc = correct / total

        # 🧪 Validation
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        print(f"Epoch {epoch:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger = 0
            torch.save(model.state_dict(), "best_fast_model.pth")
            print("✅ Best model saved!")
        else:
            trigger += 1
            if trigger >= patience:
                print("🛑 Early stopping.")
                break

    # 🧾 Final Evaluation
    model.load_state_dict(torch.load("best_fast_model.pth"))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    print("\n📈 Classification Report:")
    print(classification_report(labels, preds))
    print("🧩 Confusion Matrix:")
    sns.heatmap(confusion_matrix(labels, preds), annot=True, fmt='d')
    plt.show()
    print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.4f}")


# 🧩 Entry Point (IMPORTANT FOR WINDOWS!)
if __name__ == "__main__":
    main()
