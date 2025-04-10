from flask import Flask, request, jsonify
from flask_cors import CORS
from camera import capture_image_and_load
from machine_learning import run_test_environment
from mongo_db import (
    create_sample,
    get_samples,
    get_sample_by_id,
    get_sample_by_name,
    update_sample,
    delete_sample
)

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello from Flask"



#############################
#      Single Evaluation
#############################
@app.route("/evaluate", methods=["GET"])
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


#############################
#          MongoDB
#############################

#############################
#           CREATE
#############################
@app.route('/samples', methods=['POST'])
def route_create_sample():
    """
    Create a new Sample.
    Expects JSON in the body, for example:
    {
      "ImageName": "image.png",
      "FilePath": "c:/gallery/image.png",
      "SystemAnalysis": "Paper",
      "ImageClass": "Paper",
      "Outcome": "Null"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400
    
    try:
        sample_id = create_sample(data)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    return jsonify({"message": "Sample created", "id": sample_id}), 201

#############################
#            READ
#############################
@app.route('/samples', methods=['GET'])
def route_get_samples():
    """
    Retrieve all Sample documents.
    """
    all_samples = get_samples()
    return jsonify(all_samples), 200


@app.route('/samples/<sample_id>', methods=['GET'])
def route_get_sample_by_id(sample_id):
    """
    Retrieve a single Sample by its ID.
    """
    sample = get_sample_by_id(sample_id)
    if sample:
        return jsonify(sample), 200
    else:
        return jsonify({"error": "Sample not found"}), 404


@app.route('/samples/name/<name>', methods=['GET'])
def route_get_sample_by_name(name):
    """
    Retrieve a single Sample by its ImageName.
    """
    sample = get_sample_by_name(name)
    if sample:
        return jsonify(sample), 200
    else:
        return jsonify({"error": "Sample not found"}), 404

#############################
#           UPDATE
#############################
@app.route('/samples/<sample_id>', methods=['PUT'])
def route_update_sample(sample_id):
    """
    Update a Sample by its ID.
    Expects JSON body with any fields to update. For example:
    {
      "Outcome": "Success"
    }
    """
    update_data = request.get_json()
    if not update_data:
        return jsonify({"error": "No JSON body provided"}), 400

    try:
        modified_count = update_sample(sample_id, update_data)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    if modified_count > 0:
        return jsonify({"message": "Sample updated"}), 200
    else:
        return jsonify({"error": "Sample not found or no changes made"}), 404

#############################
#           DELETE
#############################
@app.route('/samples/<sample_id>', methods=['DELETE'])
def route_delete_sample(sample_id):
    """
    Delete a Sample by its ID.
    """
    deleted_count = delete_sample(sample_id)
    if deleted_count > 0:
        return jsonify({"message": "Sample deleted"}), 200
    else:
        return jsonify({"error": "Sample not found"}), 404

#############################
#          START
#############################
if __name__ == '__main__':
    app.run(debug=True, port=5050)
