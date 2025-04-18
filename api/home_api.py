# api/home_api.py
from flask import Blueprint, jsonify, current_app
from utils.camera import capture_image_and_load
from utils.machine_learning import run_test_environment
from utils.arduino import (
    initialize_connection,
    close_connection,
    start_motors_slow,      # “stepper start”
    stop_motors,            # “stepper stop”
    move_servo_0_to_100,    # “servo left‑to‑right”
    move_servo_100_to_0     # “servo right‑to‑left”
)
from data.mongo_db import create_sample
import os

home_bp = Blueprint("home_bp", __name__)

# ──────────────────────────────────────────────────────────────────────────────
# Initialise / close the Arduino connection exactly once
# ──────────────────────────────────────────────────────────────────────────────
@home_bp.before_app_first_request
def _open_serial():
    try:
        initialize_connection()         # utils.arduino takes defaults from its constants
        current_app.logger.info("Arduino serial connection initialised.")
    except Exception as e:
        # Log the error – routes will still run but commands will error out
        current_app.logger.error(f"Arduino init failed: {e}")

@home_bp.teardown_appcontext
def _close_serial(exc):
    # Always attempt clean shutdown
    try:
        close_connection()
        current_app.logger.info("Arduino serial connection closed.")
    except Exception:
        pass
# ──────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────
# 1.  Stepper‑motor routes
# ─────────────────────────────
@home_bp.route("/start_stepper", methods=["POST"])
def start_stepper_route():
    """
    Starts the conveyor / stepper motors (slow forward).
    """
    try:
        start_motors_slow()
        return jsonify({"message": "Stepper started"}), 200
    except Exception as ex:
        return jsonify({"error": f"Stepper start failed: {ex}"}), 500


@home_bp.route("/stop_stepper", methods=["POST"])
def stop_stepper_route():
    """
    Stops the conveyor / stepper motors immediately.
    """
    try:
        stop_motors()
        return jsonify({"message": "Stepper stopped"}), 200
    except Exception as ex:
        return jsonify({"error": f"Stepper stop failed: {ex}"}), 500

# ─────────────────────────────
# 2.  Servo routes
# ─────────────────────────────
@home_bp.route("/servo_ltr", methods=["POST"])
def servo_left_to_right_route():
    """
    Rotate the sorting arm: **Left → Right** (0° → 100°).
    """
    try:
        move_servo_0_to_100()
        return jsonify({"message": "Servo moved left‑to‑right"}), 200
    except Exception as ex:
        return jsonify({"error": f"Servo LTR failed: {ex}"}), 500


@home_bp.route("/servo_rtl", methods=["POST"])
def servo_right_to_left_route():
    """
    Rotate the sorting arm: **Right → Left** (100° → 0°).
    """
    try:
        move_servo_100_to_0()
        return jsonify({"message": "Servo moved right‑to‑left"}), 200
    except Exception as ex:
        return jsonify({"error": f"Servo RTL failed: {ex}"}), 500

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
    save_dir  = r"C:\Users\user\OneDrive\Gallery\Pictures"
    filename  = "camera_image.jpg"
    threshold = 0.7

    # 1. Capture image
    try:
        saved_path, pil_img = capture_image_and_load(save_dir, filename)
    except Exception as e:
        return jsonify({"error": f"Camera capture failed: {e}"}), 500

    # 2. Model inference
    try:
        label, confidence_str = run_test_environment(threshold, pil_img)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    # 3. Build & insert sample
    image_name       = os.path.basename(saved_path)
    system_analysis  = label
    outcome          = "Failure" if system_analysis == "Uncertain" else None

    sample_doc = {
        "image_name":            image_name,
        "file_path":             saved_path,
        "system_analysis":       system_analysis,
        "image_class":           None,
        "outcome":               outcome,
        "confidence_percentage": confidence_str
    }

    try:
        inserted_id = create_sample(sample_doc)
    except ValueError as ve:
        return jsonify({"error": f"Validation error: {ve}"}), 400
    except Exception as ex:
        return jsonify({"error": f"Database error: {ex}"}), 500

    # 4. Respond to client
    return jsonify({
        "message":     "Inference complete; sample created",
        "file_path":   saved_path,
        "label":       label,
        "confidence":  confidence_str,
        "inserted_id": inserted_id
    }), 200
