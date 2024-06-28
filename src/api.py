import argparse
import torch
from torchvision import transforms
from PIL import Image
from model import PneumoniaDetectionModel

from fastapi import FastAPI
from typing import Annotated
from fastapi import FastAPI, UploadFile
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model(model_path):
    model = PneumoniaDetectionModel(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

model_path = 'models/pneumonia_detection_model_best.pth'
model = load_model(model_path)

@app.post("/is-pneumonia")
async def is_pneumonia(file: UploadFile):
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content))
    image_tensor = preprocess_image(img)
    prediction = predict(model, image_tensor)
    return {"result": bool(prediction)}

