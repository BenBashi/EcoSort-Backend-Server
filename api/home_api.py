# api/home_api.py
from flask import Blueprint, jsonify, current_app, request
from utils.camera import capture_image_and_load
from utils.machine_learning import run_test_environment
from utils.arduino import (
    initialize_connection,
    close_connection,
    start_motors_slow,      # “stepper start”
    stop_motors,            # “stepper stop”
    push_right,
    push_left
)
from data.mongo_db import create_sample
import uuid
import os, atexit, time

home_bp = Blueprint("home_bp", __name__)
# ──────────────────────────────────────────────────────────────────────────────
# Initialise / close the Arduino connection exactly once
# ──────────────────────────────────────────────────────────────────────────────
@home_bp.record_once
def _init_serial(state):
    """Runs exactly once, when the blueprint is registered."""
    try:
        initialize_connection()
        state.app.logger.info("Arduino serial connection initialised.")
    except Exception as e:
        state.app.logger.error(f"Arduino init failed: {e}")
# ──────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────
# 1.  Stepper‑motor routes
# ─────────────────────────────
@home_bp.route("/system_start", methods=["POST"])
def system_start_route():
    """
    Starts the conveyor / stepper motors (slow forward).
    """
    try:
        start_motors_slow()
        return jsonify({"message": "Stepper started"}), 200
    except Exception as ex:
        return jsonify({"error": f"Stepper start failed: {ex}"}), 500
        

@home_bp.route("/system_stop", methods=["POST"])
def system_stop_route():
    """
    Stops the conveyor / stepper motors immediately.
    """
    try:
        stop_motors()
        return jsonify({"message": "Stepper stopped"}), 200
    except Exception as ex:
        return jsonify({"error": f"Stepper stop failed: {ex}"}), 500

SERVO_ACTIONS = {
    "Paper": push_right,
    "Plastic": push_left
}

# ─────────────────────────────
# 2.  Servo routes
# ─────────────────────────────

@home_bp.route("/servo_push", methods=["POST"])
def servo_push_route():
    """
    Move the servo based on waste class sent in POST body:
    - Paper: push right
    - Plastic: push left
    - Other: no action
    """
    data = request.get_json()
    label = data.get("trueClass")

    if not label:
        return jsonify({"error": "Missing 'label' in request body"}), 400

    if label in ("Other", "Track"):
        return jsonify({"message": "Waste product is 'Other' or 'None'; no servo action taken."}), 200

    action = SERVO_ACTIONS.get(label)
    if not action:
        return jsonify({"error": f"Unknown label: {label}"}), 400

    try:
        time.sleep(1.7)  # wait until the product reaces the end of the belt
        action()  # push_right() or push_left()
        return jsonify({"message": f"Servo pushed for {label}"}), 200
    except Exception as ex:
        return jsonify({"error": f"Servo push failed: {ex}"}), 500

# ─────────────────────────────
# 3.  Image‑capture / prediction route
# ─────────────────────────────

@home_bp.route("/evaluate", methods=["POST"])
def evaluate_route():
    """
    • Capture an image
    • Run model inference
    • Save a new sample to MongoDB
    • Return prediction + DB id
    """
    threshold = float(current_app.config.get("PREDICTION_THRESHOLD", 70))

    try:
        os.makedirs("images", exist_ok=True)
        filename = f"{uuid.uuid4().hex[:8]}.jpg"
        saved_path, pil_img = capture_image_and_load("images", filename)
        current_app.logger.info(f"Image saved: {saved_path}")
    except Exception as e:
        return jsonify({"error": f"Camera error: {e}"}), 500

    try:
        label, confidence_str = run_test_environment(pil_img)
        current_app.logger.info(f"Prediction: {label} ({confidence_str})")
    except Exception as e:
        return jsonify({"error": f"Model error: {e}"}), 500

    if float(confidence_str) > threshold and label in SERVO_ACTIONS:
        try:
            time.sleep(1.7)  # wait until the product reaces the end of the belt
            SERVO_ACTIONS[label]()  # Actuate servo
            start_motors_slow()
        except Exception as e:
            return jsonify({"error": f"Hardware action failed: {e}"}), 500

    # Skip saving to the DB if the label is "Track"
    if label != 'Track':
        try:
            inserted_id = create_sample({
                "image_name": os.path.basename(saved_path),
                "file_path": saved_path,
                "system_analysis": label,
                "image_class": None,
                "confidence_percentage": confidence_str
            })
        except Exception as e:
            return jsonify({"error": f"DB error: {e}"}), 500
    else:
        inserted_id = None  # No DB insertion for 'Track' label

    return jsonify({
        "message": "Success",
        "image_name": filename,
        "file_path": saved_path,
        "label": label,
        "confidence": confidence_str,
        "inserted_id": inserted_id
    }), 200

atexit.register(close_connection)