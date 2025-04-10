import cv2
import os
from PIL import Image

def capture_image_and_load(save_dir, filename="captured_image.jpg"):
    """
    Captures a single frame from the default camera (index=0),
    saves it to 'save_dir/filename',
    and loads it as a PIL Image in RGB mode.

    Returns:
      (save_path, pil_img)
        - save_path (str): The full path to the saved file
        - pil_img (PIL.Image.Image): The loaded image in RGB
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera (index 0). Check if the device is connected/accessible.")

    # Capture one frame
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture an image from the camera.")

    # Save to disk (frame is in BGR format)
    cv2.imwrite(save_path, frame)

    # Load that saved file as a PIL Image (convert BGR->RGB automatically by PIL)
    pil_img = Image.open(save_path).convert("RGB")

    return save_path, pil_img
