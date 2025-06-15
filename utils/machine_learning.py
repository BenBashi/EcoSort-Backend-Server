import os
import shutil
import random
import gdown
import zipfile
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
import csv  # Add import for CSV file creation


# Load Configuration
load_dotenv(dotenv_path="./.env.local")
MODEL_URL = os.environ.get("MODEL_URL", "")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
KAGGLE_DATASET_URL = os.environ.get("KAGGLE_DATASET_URL")
TRACK_IMAGES_URL = os.environ.get("TRACK_IMAGES_URL") 
UNCERTAIN_DIR = "./images/low_confidence"
ORIGINAL_DATASET_DIR = "./original_dataset"
BALANCED_DATASET_DIR = "./balanced_fewshot_dataset"
RETRAINED_MODEL_PATH = "./resnet50_recycling_retrained.pth"
CLASSES = ["Plastic", "Paper", "Other", "Track"]

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
    return transforms.Compose([
        transforms.Resize((224, 224)),
        RandAugment(num_ops=2, magnitude=5),
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
# Load & Balance Few-Shot Data from uncertain images + original_dataset
# -----------------------------------------------------------------------------
def prepare_balanced_fewshot_dataset(uncertain_root, filler_root, transform, output_csv_path="./balanced_dataset_labels.csv"):
    """
    Ensures all classes have the same number of samples (equal to the max).
    Creates a CSV file with image paths and labels.
    Returns a balanced Subset and class_to_idx.
    If no uncertain images exist at all, returns (None, None).
    """
    # Setup Kaggle API credentials
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
        f.write(f'{{"username":"{KAGGLE_USERNAME}","key":"{KAGGLE_KEY}"}}')
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

    # Download and unzip dataset if not present
    if not os.path.exists("./original_dataset") or not os.listdir("./original_dataset"):
        print("Downloading from Kaggle...")
        subprocess.run([
            "kaggle", "datasets", "download", "-d", KAGGLE_DATASET_URL,
            "-p", ORIGINAL_DATASET_DIR, "--unzip"
        ], check=True)
    
    # Add Track photos that are not in the Kaggle dataset
    if not os.path.exists("./original_dataset/train/track") or not os.listdir("./original_dataset/train/track"):
        print("Adding Track images to original dataset...")
        gdown.download(TRACK_IMAGES_URL, "./original_dataset/train/track.zip", quiet=False)
        with zipfile.ZipFile("./original_dataset/train/track.zip", "r") as zip_ref:
            zip_ref.extractall("./original_dataset/train/")
        os.remove("./original_dataset/train/track.zip")

    # Prepare uncertain dataset class mapping manually (handle missing folders)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(["plastic", "paper", "other", "track"])}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    image_paths_by_class = defaultdict(list)
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

    for cls_name in class_to_idx:
        class_dir = os.path.join(uncertain_root, cls_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Missing directory for class '{cls_name}' in uncertain_root.")
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(supported_extensions):
                image_paths_by_class[cls_name].append(os.path.join(class_dir, fname))

    # Abort early if no uncertain images at all
    total_images = sum(len(paths) for paths in image_paths_by_class.values())
    if total_images == 0:
        print("No images found in uncertain images path.")
        return None, None

    # Determine max class size
    target_per_class = max(50, max(len(paths) for paths in image_paths_by_class.values()))

    # Fill all classes up to max_samples using filler
    final_images = []
    final_labels = []

    for cls_name, label_id in class_to_idx.items():
        paths = image_paths_by_class.get(cls_name, [])
        needed = target_per_class - len(paths)

        final_images.extend(paths)
        final_labels.extend([label_id] * len(paths))

        if needed > 0:
            filler_dir = os.path.join(filler_root, cls_name)
            if not os.path.isdir(filler_dir):
                print(f"Warning: Missing filler directory for class '{cls_name}' in filler_root.")
                continue

            filler_candidates = [
                os.path.join(filler_dir, f)
                for f in os.listdir(filler_dir)
                if f.lower().endswith(supported_extensions)
            ]
            print(f"Found {len(filler_candidates)} filler images for class '{cls_name}'.")

            random.shuffle(filler_candidates)
            final_images.extend(filler_candidates[:needed])
            final_labels.extend([label_id] * min(needed, len(filler_candidates)))

            if len(filler_candidates) < needed:
                print(f"Warning: Insufficient filler images for class '{cls_name}'. Needed: {needed}, Found: {len(filler_candidates)}")

    # Create CSV file with image paths and labels
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "label"])
        for img_path, label in zip(final_images, final_labels):
            writer.writerow([img_path, idx_to_class[label]])
    print(f"✅ CSV file created at: {output_csv_path}")

    # Wrap in a custom dataset to support Subset
    from torchvision.datasets.folder import default_loader
    class FewShotImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, labels, transform):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = default_loader(self.image_paths[idx])
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]

    balanced_dataset = FewShotImageDataset(final_images, final_labels, transform)
    return balanced_dataset, class_to_idx


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

    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Prepare dataset
    transform = get_augmented_transform()
    balanced_subset, class_to_idx = prepare_balanced_fewshot_dataset(
        uncertain_root=uncertain_root,
        filler_root=filler_root,
        transform=transform
    )

    # If no data is available, skip training
    if balanced_subset is None:
        print("Retraining skipped: No images found in uncertain_root.")
        return

    loader = DataLoader(balanced_subset, batch_size=8, shuffle=True)

    # Retrain
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    epochs = 5

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(loader)
        epoch_accuracy = 100 * correct / total
        print(f"📊 Epoch {epoch+1} - Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")

    # Save updated model
    torch.save(model.state_dict(), output_weights_path)
    print(f"✅ Retrained model saved at: {output_weights_path}")
