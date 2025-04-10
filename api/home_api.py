# api/home_api.py
from flask import Blueprint, request, jsonify
from utils.camera import capture_image_and_load
from utils.machine_learning import run_test_environment
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


@home_bp.route("/evaluate", methods=["GET"])
def evaluate_route():
    """
    1) Capture a photo from the camera -> PIL image
    2) Run the model inference with 'run_test_environment'
    3) Return the predicted label & confidence
    """

    # Directory & filename for the camera capture
    save_dir = r"C:\Users\user\OneDrive\Gallery\Pictures"
    filename = "camera_image.jpg"
    threshold = float(0.7)

    try:
        # 1) Capture + load as a PIL image
        saved_path, pil_img = capture_image_and_load(save_dir, filename)
    except Exception as e:
        return jsonify({"error": f"Camera capture failed: {e}"}), 500

    try:
        # 2) Run model inference
        label, confidence_str = run_test_environment(threshold, pil_img)
        # 3) Return JSON
        return jsonify({
            "message": "Inference complete",
            "file_path": saved_path,
            "label": label,
            "confidence": confidence_str
        }), 200
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500
