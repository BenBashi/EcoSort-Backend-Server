import os
import shutil
import random
import gdown
import torch
import subprocess
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.transforms import RandAugment, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from collections import defaultdict


# Load Configuration
load_dotenv(dotenv_path="./.env.local")
MODEL_URL = os.environ.get("MODEL_URL", "")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
KAGGLE_DATASET_URL = os.environ.get("KAGGLE_DATASET_URL")
UNCERTAIN_DIR = "./images/low_confidence"
ORIGINAL_DATASET_DIR = "./original_dataset"
BALANCED_DATASET_DIR = "./balanced_fewshot_dataset"
RETRAINED_MODEL_PATH = "./resnet50_recycling_retrained.pth"
CLASSES = ["Plastic", "Paper", "Other", "Track"]
MIN_IMAGES_PER_CLASS = 5  # few-shot target

# -----------------------------------------------------------------------------
# Download the model file ONCE (if it doesn't exist yet)
# -----------------------------------------------------------------------------
model_path_default = "./resnet50_recycling_adjusted.pth"
if not os.path.exists(model_path_default):
    gdown.download(MODEL_URL, model_path_default, quiet=False)


# -----------------------------------------------------------------------------
# Model creation & loading
# -----------------------------------------------------------------------------
def create_model():
    """
    Create a ResNet50 model:
      - freeze all layers except layer4
      - replace final FC layer with a 4-class output
    """
    model = models.resnet50(pretrained=True)

    # Freeze most layers
    for param in model.parameters():
        param.requires_grad = False

    # Keep layer4 trainable
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final classifier with a 4-class output
    model.fc = nn.Linear(model.fc.in_features, 4)
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
# Transform & Prediction
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

def predict_recycling_class(pil_img, model, device, transform):
    """
    Predict the recycling class for a given PIL image.

    Returns:
      (predicted_idx, confidence, class_confidences)
        - predicted_idx (int) : index of predicted class
        - confidence (float)  : highest confidence
        - class_confidences (np.array): all class probabilities
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

    return predicted_idx, confidence, class_confidences


# -----------------------------------------------------------------------------
# Single Image Classification Utility
# -----------------------------------------------------------------------------
def run_test_environment(pil_img):
    """
    - Loads the model from 'model_path'
    - Classifies the given PIL image
    - Returns (label, confidence_str)
    """

    model, device = load_model_weights(model_path_default)
    transform = get_transform()

    predicted_idx, confidence, _ = predict_recycling_class(
        pil_img, model, device, transform
    )

    label = CLASSES[predicted_idx]
    confidence_str = f"{confidence * 100:.2f}"
    
    return label, confidence_str

# -----------------------------------------------------------------------------
# Strong Transform with RandAugment For Retraining
# -----------------------------------------------------------------------------
def get_augmented_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        RandAugment(num_ops=2, magnitude=9),  # Strong augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# -----------------------------------------------------------------------------
# Load & Balance Few-Shot Data from uncertain_images + original_dataset
# -----------------------------------------------------------------------------
def prepare_balanced_fewshot_dataset(uncertain_root, filler_root, transform):
    """
    Ensures all classes have the same number of samples (equal to the max).
    Returns a balanced Subset and class_to_idx.
    """

    # Make sure Kaggle directory and API token are set up
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
        f.write(f'{{"username":"{KAGGLE_USERNAME}","key":"{KAGGLE_KEY}"}}')

    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

    # Download and unzip dataset
    subprocess.run([
    "kaggle", "datasets", "download", "-d", KAGGLE_DATASET_URL, "-p", ORIGINAL_DATASET_DIR, "--unzip"],
    check=True)
    
    # Load uncertain dataset
    uncertain_dataset = datasets.ImageFolder(uncertain_root, transform=transform)
    class_to_idx = uncertain_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Group uncertain images by class
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(uncertain_dataset):
        class_indices[label].append(idx)
    
    # Determine target count (maximum class size)
    max_samples = max(len(indices) for indices in class_indices.values())
    
    # Add filler images from original dataset if needed
    all_indices = []
    for class_id, indices in class_indices.items():
        needed = max_samples - len(indices)
        all_indices.extend(indices)

        if needed > 0:
            class_name = idx_to_class[class_id]
            filler_path = os.path.join(filler_root, class_name)
            filler_dataset = datasets.ImageFolder(filler_root, transform=transform)
            
            # Filter only relevant class
            class_label = filler_dataset.class_to_idx[class_name]
            class_indices_filler = [i for i, (_, lbl) in enumerate(filler_dataset) if lbl == class_label]
            random.shuffle(class_indices_filler)
            all_indices.extend(class_indices_filler[:needed])
    
    balanced_subset = Subset(uncertain_dataset, all_indices)
    return balanced_subset, class_to_idx


# -----------------------------------------------------------------------------
# Retrain on few-shot balanced dataset
# -----------------------------------------------------------------------------
def retrain_fewshot_model(uncertain_root, filler_root, model_weights_path, output_weights_path):
    """
    Retrains your model on a few-shot balanced dataset using strong augmentations.
    Saves the updated model weights to output_weights_path.
    """
    # Load model
    model = create_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)

    # Freeze everything except final FC
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    # Prepare dataset
    transform = get_augmented_transform()
    balanced_subset, class_to_idx = prepare_balanced_fewshot_dataset(
        uncertain_root=uncertain_root,
        filler_root=filler_root,
        transform=transform
    )
    loader = DataLoader(balanced_subset, batch_size=8, shuffle=True)

    # Retrain
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)
    epochs = 5

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {running_loss:.4f}")

    # Save updated model
    torch.save(model.state_dict(), output_weights_path)
    print(f"✅ Retrained model saved at: {output_weights_path}")