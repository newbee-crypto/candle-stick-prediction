# predict.py

import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from model import Model   # import model

#  Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Load model
model = Model().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

print(" Model loaded")

#  Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#  Load image
image_path = "D:/full-stack2/data/charts/GOOGL/2025-11-18.png"

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

#  Input features (change these)
features = torch.tensor([[60.97, 8.4961, 1.9244]], dtype=torch.float32).to(device)

#  Prediction
with torch.no_grad():
    output = model(image, features)
    pred = torch.argmax(output, dim=1).item()

    probs = F.softmax(output, dim=1)

#  Label mapping
labels = ["Downtrend", "Neutral", "Uptrend"]

print(f"Prediction: {labels[pred]}")
print(f"Confidence: {probs}")