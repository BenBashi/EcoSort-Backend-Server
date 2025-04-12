# api/dashboard_api.py
from flask import Blueprint, request, jsonify
from data.mongo_db import samples_collection, update_sample, get_samples, delete_sample
from bson.objectid import ObjectId

dashboard_bp = Blueprint("dashboard_bp", __name__)

@dashboard_bp.route("/update_result", methods=["POST"])
def update_result_route():
    """
    Allows users to update the image_class field for a sample.
    Expects JSON with:
      {
        "sample_id": <the Mongo _id string>,
        "system_analysis": <the model's predicted label, e.g. "Paper" or "Plastic">,
        "image_class": <the user's manual classification>
      }

    Sets outcome="Success" if image_class == system_analysis, else "Failure".
    """
    data = request.json
    sample_id = data.get("sample_id")
    system_analysis = data.get("system_analysis")
    image_class = data.get("image_class")

    # Validate input presence
    if not sample_id or not system_analysis or not image_class:
        return jsonify({"error": "Missing one of: sample_id, system_analysis, image_class"}), 400

    # Determine outcome
    outcome = "Success" if (image_class == system_analysis) else "Failure"

    # Prepare partial update
    update_data = {
        "image_class": image_class,
        "outcome": outcome
    }

    # Attempt to update the DB
    try:
        modified_count = update_sample(sample_id, update_data)
        return jsonify({"modified_count": modified_count}), 200
    except ValueError as ve:
        return jsonify({"error": f"Validation error: {ve}"}), 400
    except Exception as ex:
        return jsonify({"error": f"Database error: {ex}"}), 500


@dashboard_bp.route("/get_results", methods=["GET"])
def get_results_route():
    """
    Retrieves all samples from the database.
    """
    # You can directly call your mongo_db helper
    results = get_samples()
    return jsonify({"samples": results}), 200


@dashboard_bp.route("/calculate_accuracy", methods=["GET"])
def calculate_accuracy_route():
    """
    Compares 'SystemAnalysis' vs. 'ImageClass' to compute % correct.
    """
    docs = get_samples()
    if not docs:
        return jsonify({"accuracy": 0}), 200

    total = len(docs)
    correct = 0
    for doc in docs:
        if doc.get("SystemAnalysis") == doc.get("ImageClass"):
            correct += 1

    accuracy = (correct / total) * 100
    return jsonify({"accuracy": accuracy}), 200


@dashboard_bp.route("/default_model", methods=["POST"])
def default_model_route():
    """
    Resets the active classifier to the default pre-trained model.
    (Call your internal function if it exists)
    """
    # e.g., set_default_model()
    # For now, just a placeholder response
    return jsonify({"message": "Default model reactivated"}), 200


@dashboard_bp.route("/retrain", methods=["POST"])
def retrain_route():
    """
    Retrains the model with updated/mislabeled data.
    (Call your internal function if it exists)
    """
    # e.g., retrain_model()
    return jsonify({"message": "Retraining started..."}), 200


@dashboard_bp.route("/delete_sample", methods=["POST"])
def delete_sample_route():
    """
    Deletes a Sample document by _id.
    Expects JSON with {"sample_id": <the Mongo _id string>}.
    Returns JSON with {"deleted_count": <number_of_docs_deleted>}.
    """
    data = request.json
    sample_id = data.get("sample_id")

    if not sample_id:
        return jsonify({"error": "Missing sample_id"}), 400

    try:
        deleted_count = delete_sample(sample_id)
        return jsonify({"deleted_count": deleted_count}), 200
    except Exception as ex:
        return jsonify({"error": f"Database error: {ex}"}), 500