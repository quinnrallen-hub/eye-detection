#!/usr/bin/env python3
"""
Predict if eyes are open or closed in an image.
Usage: python predict.py <image_path>
"""

import sys
import cv2
import torch
import torch.nn as nn
import numpy as np

IMG_SIZE = 64

class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def predict(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EyeCNN().to(device)
    model.load_state_dict(torch.load(
        "/home/quinn/eye_detection_project/eye_classifier.pth",
        map_location=device,
        weights_only=True
    ))
    model.eval()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred = outputs.argmax(dim=1).item()

    labels = ['OPEN', 'CLOSED']
    confidence = probs[0][pred].item() * 100

    print(f"Eyes: {labels[pred]} ({confidence:.1f}% confidence)")
    return labels[pred], confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    predict(sys.argv[1])
