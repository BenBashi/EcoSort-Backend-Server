import os
import gdown
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv

# Load Configuration
load_dotenv(dotenv_path="./.env.local")
MODEL_URL = os.environ.get("MODEL_URL", "")

# -----------------------------------------------------------------------------
# 1. Download the model file ONCE (if it doesn't exist yet)
# -----------------------------------------------------------------------------
model_path_default = "resnet50_recycling.pth"
if not os.path.exists(model_path_default):
    gdown.download(MODEL_URL, model_path_default, quiet=False)


# -----------------------------------------------------------------------------
# 2. Model creation & loading
# -----------------------------------------------------------------------------
def create_model():
    """
    Create a ResNet50 model:
      - freeze all layers except layer4
      - replace final FC layer with a 3-class output
    """
    model = models.resnet50(pretrained=True)

    # Freeze most layers
    for param in model.parameters():
        param.requires_grad = False

    # Keep layer4 trainable
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final classifier with a 3-class output
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model

def load_model_weights(model_path):
    """
    Loads model weights from 'model_path', moves model to device (CPU or CUDA).
    Returns (model, device).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device


# -----------------------------------------------------------------------------
# 3. Transform & Prediction
# -----------------------------------------------------------------------------
def get_transform():
    """
    Standard transform: resize to 224x224, convert to tensor, normalize with
    ImageNet means & std.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def predict_recycling_class(pil_img, model, device, transform, threshold=0.7):
    """
    Predict the recycling class for a given PIL image.
      - threshold: if max confidence < threshold => 'uncertain'

    Returns:
      (predicted_idx, confidence, class_confidences, is_uncertain)
        - predicted_idx (int) : index of predicted class
        - confidence (float)  : highest confidence
        - class_confidences (np.array): all class probabilities
        - is_uncertain (bool) : True if confidence < threshold
    """
    # Preprocess: PIL -> tensor -> device
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        max_prob, predicted = torch.max(probabilities, 0)

    predicted_idx = predicted.item()
    confidence = max_prob.item()
    class_confidences = probabilities.cpu().numpy()

    is_uncertain = (confidence < threshold)
    return predicted_idx, confidence, class_confidences, is_uncertain


# -----------------------------------------------------------------------------
# 4. Single Image Classification Utility
# -----------------------------------------------------------------------------
def run_test_environment(threshold, pil_img, model_path=model_path_default):
    """
    - Loads the model from 'model_path'
    - Classifies the given PIL image
    - Returns (label, confidence_str)
    """
    class_names = ["Plastic", "Paper", "Other"]

    model, device = load_model_weights(model_path)
    transform = get_transform()

    predicted_idx, confidence, _, is_uncertain = predict_recycling_class(
        pil_img, model, device, transform, threshold
    )

    # Determine label
    if is_uncertain:
        label = "Uncertain"
    else:
        label = class_names[predicted_idx]

    confidence_str = f"{confidence * 100:.2f}"
    return label, confidence_str
