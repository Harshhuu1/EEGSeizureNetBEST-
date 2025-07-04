from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
from torchvision import transforms
from io import BytesIO

app = FastAPI()

# Load model
try:
    model = torch.jit.load("seizure_model_efficientnet.pt", map_location="cpu")
    model.eval()
    print("✅ Model loaded.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/")
async def root():
    return {"message": "API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(output, dim=1).item()
            label = "interictal" if pred == 0 else "preictal"

        return {"prediction": label}

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
