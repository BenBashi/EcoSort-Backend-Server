from flask import Blueprint, request, jsonify
from utils.machine_learning import (
    create_model, 
    get_augmented_transform, 
    prepare_balanced_fewshot_dataset, 
    retrain_fewshot_model
)
from data.mongo_db import (
    update_sample,
    get_samples,
    delete_sample,
    get_sample_by_id,
    delete_all_samples
)

dashboard_bp = Blueprint("dashboard_bp", __name__)

@dashboard_bp.route("/results/<sample_id>", methods=["PUT"])
def update_result_route(sample_id):
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
    system_analysis = data.get("systemAnalysis")
    image_class = data.get("trueClass")

    if not sample_id or not system_analysis or not image_class:
        return jsonify({"error": "Missing one of: sample_id, system_analysis, image_class"}), 400

    outcome = "Success" if (image_class == system_analysis) else "Failure"
    update_data = {
        "image_class": image_class,
        "outcome": outcome
    }

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
    try:
        results = get_samples()
        return jsonify({"samples": results}), 200
    except Exception as ex:
        return jsonify({"error": f"Database error: {ex}"}), 500

@dashboard_bp.route("/results/<sample_id>", methods=["GET"])
def get_result_route(sample_id):
    """
    Returns a single Sample document by _id.
    Expects JSON with:
      { "sample_id": <the Mongo _id string> }
    If found, returns the doc. If not found, returns None.
    """
    if not sample_id:
        return jsonify({"error": "Missing sample_id"}), 400

    try:
        doc = get_sample_by_id(sample_id)
        # doc will be None if not found
        return jsonify({"sample": doc}), 200
    except Exception as ex:
        return jsonify({"error": f"Database error: {ex}"}), 500

@dashboard_bp.route("/calculate_accuracy", methods=["GET"])
def calculate_accuracy_route():
    """
    Compares 'system_analysis' vs. 'image_class' to compute % correct.
    """
    try:
        docs = get_samples()
    except Exception as ex:
        return jsonify({"error": f"Database error: {ex}"}), 500

    if not docs:
        return jsonify({"accuracy": 0}), 200

    total = len(docs)
    correct = 0
    for doc in docs:
        # Adjust keys if your DB fields differ in naming
        if doc.get("image_class") in [ doc.get("system_analysis"), None ]:
            correct += 1

    accuracy = (correct / total) * 100
    return jsonify({"accuracy": accuracy}), 200

@dashboard_bp.route("/default_model", methods=["POST"])
def default_model_route():
    """
    Resets the active classifier to the default pre-trained model.
    (Placeholder)
    """
    return jsonify({"message": "Default model reactivated"}), 200

@dashboard_bp.route("/retrain", methods=["POST"])
def retrain_endpoint():
    retrain_fewshot_model(
        uncertain_root="./images/low_confidence",
        filler_root="./original_dataset/train/",
        model_weights_path="./resnet50_recycling_adjusted.pth",
        output_weights_path="./resnet50_recycling_retrained.pth"
    )
    return jsonify({
        "status": "Success", 
        "message": "Retraining complete"
    }), 200

@dashboard_bp.route("/delete_result", methods=["POST"])
def delete_result_route():
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

# A route to delete all samples from the database
@dashboard_bp.route("/samples", methods=["DELETE"])
def delete_all_results_route():
    """
    Deletes all samples from the database.
    """
    try:
        deleted_count = delete_all_samples()
        return jsonify({"deleted_count": deleted_count}), 200
    except Exception as ex:
        return jsonify({"error": f"Database error: {ex}"}), 500
