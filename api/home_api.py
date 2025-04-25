# api/home_api.py
from flask import Blueprint, request, jsonify
from utils.camera import capture_image_and_load
from utils.machine_learning import run_test_environment
from data.mongo_db import create_sample
import os
import uuid


# For treadmill, camera, classification, etc., import from your utils or data
# from utils.arduino import start_motors_slow, stop_motors
# from data.mongo_db import create_sample, ...

home_bp = Blueprint("home_bp", __name__)

@home_bp.route("/system_start", methods=["POST"])
def system_start_route():
    """
    Starts the system:
      1) Activate treadmill (motors)
      2) Begin scanning loop for images
    """
    # start_motors_slow()  # if implemented
    # image_scan()         # or similar function

    return jsonify({"message": "System started"}), 200


@home_bp.route("/system_stop", methods=["POST"])
def system_stop_route():
    """
    Stops the system:
      - Stop treadmill (motors)
      - Stop scanning loop
    """
    # stop_motors()  # if implemented
    return jsonify({"message": "System stopped"}), 200


@home_bp.route("/evaluate", methods=["POST"])
def evaluate_route():
    """
    1) Capture a photo from the camera -> PIL image
    2) Run the model inference (run_test_environment)
    3) Insert a new sample into MongoDB with the inference results
    4) Return the predicted label & confidence
    """
    # Directory & filename for the camera capture
    save_dir = os.path.join(os.getcwd(), "images")
    image_uuid = str(uuid.uuid4())[:8]
    filename = f"{image_uuid}.jpg"
    threshold = 0.7  # example threshold

    # 1) Capture image
    try:
        os.makedirs(save_dir, exist_ok=True)
        saved_path, pil_img = capture_image_and_load(save_dir, filename)
    except Exception as e:
        return jsonify({"error": f"Camera capture failed: {e}"}), 500

    # 2) Run model inference
    try:
        label, confidence_str = run_test_environment(threshold, pil_img)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    # 3) Create sample in MongoDB
    #    Derive image_name from the file path, e.g. the filename part
    image_name = os.path.basename(saved_path)

    # system_analysis is the model's predicted label
    system_analysis = label

    new_sample = {
        "image_name": image_name,               # e.g. "camera_image.jpg"
        "file_path": saved_path,                # full path
        "system_analysis": system_analysis,      # from model inference
        "image_class": None,                    # user has not updated it yet
        "confidence_percentage": confidence_str  # must be a string
    }

    try:
        inserted_id = create_sample(new_sample)
    except ValueError as ve:
        return jsonify({"error": f"Validation error: {ve}"}), 400
    except Exception as ex:
        return jsonify({"error": f"Database error: {ex}"}), 500

    # 4) Return the results
    return jsonify({
        "message": "Inference complete, sample created",
        "image_name": image_name,
        "file_path": saved_path,
        "label": label,
        "confidence": confidence_str,
        "inserted_id": inserted_id
    }), 200