import cv2
import os
import time
from PIL import Image

def capture_image_and_load(save_dir, filename="captured_image.jpg"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    # 🔧 Set higher resolution (adjust based on your webcam's capability)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 📸 Warm-up: discard initial frames
    for _ in range(5):
        ret, _ = cap.read()
        time.sleep(0.1)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture image.")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Rest of your cropping logic...
    width, height = pil_img.size

    crop_width_ratio = 0.6
    crop_height_ratio = 1
    horizontal_shift_ratio = 0.05
    vertical_anchor_ratio = 0.8

    crop_width = int(width * crop_width_ratio)
    crop_height = int(height * crop_height_ratio)

    center_x = width // 2 - int(width * horizontal_shift_ratio)
    center_y = int(height * vertical_anchor_ratio)

    left = max(0, center_x - crop_width // 2)
    top = max(0, center_y - crop_height // 2)
    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)

    pil_img = pil_img.crop((left, top, right, bottom))
    pil_img.save(save_path)

    return save_path, pil_img