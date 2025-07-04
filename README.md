# 🧠 EEGSeizureNet: Efficient EEG Seizure Detection with EfficientNet-B0

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-🔥-red)
![Status](https://img.shields.io/badge/status-Active-brightgreen)

> A powerful deep learning framework for **detecting epileptic seizures** using **spectrogram images** of EEG data. Combines EfficientNet, Focal Loss, advanced augmentations, and a chatbot-based Streamlit UI for smart interaction.

---

## 🧠 Overview

EEGSeizureNet is designed to classify **preictal** (before seizure) and **interictal** (normal) brain states using spectrogram images generated from EEG `.edf` files. The system is lightweight yet powerful, making it suitable for research, clinical insights, or neuro-diagnostic prototypes.

---

## ✅ Key Features

- 🔍 **EfficientNet-B0** architecture for fast & accurate image classification
- 🔥 **Focal Loss** for improved handling of class imbalance
- 🧪 **OneCycleLR Scheduler** for faster convergence
- 🧼 **Advanced augmentations** for better generalization
- 📊 **Stratified Sampling** to maintain class balance during training
- 🛑 **Early Stopping** to prevent overfitting
- 🧠 **Streamlit UI** with integrated **chatbot** for user interaction
- 🧩 Modular code structure for **easy experimentation & deployment**

---

## 📂 Folder Structure

```
EEGSeizureNet/
│
├── dataset/
│   ├── interictal/
│   └── preictal/
│
├── generator.py       # Convert EEG .edf files to spectrogram .pngs
├── split.py           # Stratified split into training/validation sets
├── train_fast.py      # Main training script (EfficientNet + Focal Loss)
├── portable.py        # Convert .pth model to .pt for production
├── backendapi.py      # Inference API using trained model
├── app.py             # Streamlit chatbot UI
├── README.md
└── requirements.txt
```

---

## 💾 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Harshhuu1/EEGSeizureNet.git
cd EEGSeizureNet
```

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 3. Install All Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Dataset Preparation

### Step 1: Download EEG `.edf` Files

- Download [CHB-MIT EEG dataset](https://physionet.org/content/chbmit/1.0.0/) from PhysioNet.

### Step 2: Generate Spectrogram Images

Use the `generator.py` script to convert `.edf` EEG files into spectrogram `.png` images:

```bash
python generator.py
```

### Step 3: Structure Your Dataset

Organize spectrogram images into 2 folders:

```
dataset/
├── interictal/   ← Normal state
└── preictal/     ← Seizure-coming state
```

### Step 4: Split Into Train/Val Sets

```bash
python split.py
```

This script ensures class-balanced, stratified split into training and validation subsets.

---

## 🚀 Training the Model

Run the fast training pipeline with all the enhancements:

```bash
python train_fast.py
```

✔️ Trains an EfficientNet-B0 model  
✔️ Applies Focal Loss, OneCycleLR, early stopping  
✔️ Saves the best model as `best_fast_model.pth`  

---

## 🛠️ Convert to Lightweight `.pt` Model (Optional)

For faster inference or mobile deployment:

```bash
python portable.py
```

This converts your `.pth` model to `.pt` format.

---

## 🧠 Inference via Backend API

Run a local FastAPI server using:

```bash
python backendapi.py
```

- Load your `.pt` model
- Send EEG spectrogram images via API
- Get predictions in JSON format

---

## 💬 Streamlit Chatbot UI

Launch an interactive chatbot-enabled EEG classification UI:

```bash
streamlit run app.py
```

- Upload an EEG spectrogram image  
- See the model’s prediction  
- Ask follow-up questions to the chatbot about the result

---

## 🧮 Training Configuration (Default)

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)

criterion = FocalLoss(alpha=0.25, gamma=2.0)

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(train_loader),
    epochs=25
)
```

---

## 🧠 Model Architecture

```
EfficientNet-B0 (pretrained)
   ↓
Dropout + Dense(512) + ReLU
   ↓
Dropout + Dense(num_classes)
   ↓
Softmax Output (2 classes)
```

---

## 📊 Output Example

- ✅ `best_fast_model.pth` saved on best validation accuracy
- 📈 Final metrics:
  - Classification Report
  - Confusion Matrix
- 📦 Portable `.pt` model for deployment

---

## 📌 Dependencies

All packages are listed in `requirements.txt`, including:

- `torch`, `torchvision`
- `scikit-learn`, `matplotlib`, `seaborn`
- `streamlit`, `transformers`, `fastapi`

---

## 📄 License

This project is licensed under the MIT License.  
See `LICENSE` for full details.

---

## 🙋‍♂️ Author & Credits

Made with ❤️ by [Harsh Yadav](https://github.com/Harshhuu1)  
If you use this repo, consider starring 🌟 it!

