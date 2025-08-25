from flask import Blueprint, jsonify
from data.mongo_db import get_samples
from api.dashboard_api import calculate_accuracy, list_models

metrics_bp = Blueprint("metrics_bp", __name__)

@metrics_bp.route("/classification", methods=["GET"])
def get_classification_metrics():
    """
    Returns classification metrics for each class (paper, plastic, other).
    """
    try:
        docs = get_samples()
    except Exception as ex:
        return jsonify({"error": f"Database error: {ex}"}), 500

    class_names = ["paper", "plastic", "other"]
    classification = {
        "paper": {"correct": 0, "wrong_as_plastic": 0, "wrong_as_other": 0},
        "plastic": {"correct": 0, "wrong_as_paper": 0, "wrong_as_other": 0},
        "other": {"correct": 0, "wrong_as_paper": 0, "wrong_as_plastic": 0}
    }
    correct = 0

    for doc in docs:
        true_cls = (doc.get("image_class") or "").lower()
        pred_cls = (doc.get("system_analysis") or "").lower()
        if true_cls not in class_names or pred_cls not in class_names:
            continue
        if true_cls == pred_cls:
            classification[true_cls]["correct"] += 1
            correct += 1
        else:
            if pred_cls == "paper":
                classification[true_cls]["wrong_as_paper"] += 1
            elif pred_cls == "plastic":
                classification[true_cls]["wrong_as_plastic"] += 1
            elif pred_cls == "other":
                classification[true_cls]["wrong_as_other"] += 1

    total_samples = len(docs)
    accuracy = round(calculate_accuracy(), 2)

    models = list_models()
    retrain_files = [m for m in models if "retrained" in m]
    retrain_count = len(retrain_files)

    return jsonify({
        "total_samples": total_samples,
        "accuracy": accuracy,
        "retrain_count": retrain_count,
        "classification": classification
    }), 200
