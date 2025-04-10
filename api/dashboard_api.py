# api/dashboard_api.py
from flask import Blueprint, request, jsonify
from data.mongo_db import samples_collection, update_sample, get_samples
from bson.objectid import ObjectId

dashboard_bp = Blueprint("dashboard_bp", __name__)

@dashboard_bp.route("/update_result", methods=["POST"])
def update_result_route():
    """
    Allows users to update the ImageClass field for a sample.
    Expects JSON with {'sample_id': ..., 'new_class': ...}
    """
    data = request.json
    sample_id = data.get("sample_id")
    new_class = data.get("new_class")

    if not sample_id or not new_class:
        return jsonify({"error": "Missing sample_id or new_class"}), 400

    modified_count = update_sample(sample_id, {"ImageClass": new_class})
    return jsonify({"modified_count": modified_count}), 200


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
