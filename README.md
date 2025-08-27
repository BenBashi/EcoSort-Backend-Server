# EcoSort Backend Server

The *EcoSort Backend Server* is the backend component of the EcoSort system, designed to manage machine learning models, handle image classification, and interact with hardware components such as motors and servos. It also provides APIs for managing data, retraining models, and monitoring system performance.

---

## Features

•⁠  ⁠*Image Classification*: Classifies waste images into categories like Plastic, Paper, Other, and Track.

•⁠  ⁠*Model Management*: Supports model switching, retraining, and versioning.

•⁠  ⁠*Database Integration*: Stores classification results and user feedback in MongoDB.

•⁠  ⁠*Hardware Control*: Interfaces with motors and servos for waste sorting.

•⁠  ⁠*RESTful APIs*: Provides endpoints for system operations, data management, and model handling.

---

## Prerequisites

Before setting up the server, ensure you have the following installed:

•⁠  ⁠Python 3.8+

•⁠  ⁠MongoDB

•⁠  ⁠Kaggle CLI (for dataset downloads)

•⁠  ⁠CUDA (optional, for GPU acceleration)

---

## Installation

1.⁠ ⁠*Clone the Repository*:
    ```
    git clone https://github.com/your-repo/EcoSort-Backend-Server.git
    
    cd EcoSort-Backend-Server
    ⁠```


2.⁠ ⁠*Set Up a Virtual Environment*:
    ```
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ⁠```

3.⁠ ⁠*Install Dependencies*:
    ⁠```
    pip install -r requirements.txt
    ⁠```

4.⁠ ⁠*Set Up Environment Variables*:
   Create a ⁠ .env.local ⁠ file in the root directory and configure the following:
   
⁠    MODEL_URL=<URL to download the default model>
   KAGGLE_USERNAME=<Your Kaggle username>
   KAGGLE_KEY=<Your Kaggle API key>
   KAGGLE_DATASET_URL=<Kaggle dataset URL>
   TRACK_IMAGES_URL=<URL for additional Track images>
    ⁠

5.⁠ ⁠*Download the Default Model*:
   The server will automatically download the default model if it doesn't exist.

6.⁠ ⁠*Set Up MongoDB*:
   Ensure MongoDB is running locally or configure the connection string in the database utility.

---

## Usage

1.⁠ ⁠*Start the Server*:
   ⁠ bash
   flask run
    ⁠

2.⁠ ⁠*API Endpoints*:
   - *Home API*:
     - ⁠ /evaluate ⁠ (POST): Classify an image and return the result.
     - ⁠ /system_start ⁠ (POST): Start the conveyor motors.
     - ⁠ /system_stop ⁠ (POST): Stop the conveyor motors.
   - *Dashboard API*:
     - ⁠ /retrain ⁠ (POST): Retrain the model with new data.
     - ⁠ /models ⁠ (GET): List available models.
     - ⁠ /switch_model ⁠ (POST): Switch to a different model.
     - ⁠ /calculate_accuracy ⁠ (GET): Calculate the system's classification accuracy.
     - ⁠ /delete_result ⁠ (POST): Delete a specific classification result.
     - ⁠ /samples ⁠ (DELETE): Delete all classification results.

3.⁠ ⁠*Access the Frontend*:
   The backend is designed to work with the EcoSort frontend. Ensure the frontend is configured to communicate with the backend server. 
   https://github.com/DevOpsNtmm/EcoSort-Frontend

---

## Model Management

•⁠  ⁠*Model Versioning*:
  Retrained models are saved with versioned filenames (e.g., ⁠ resnet50_recycling_retrained-1.pth ⁠, ⁠ resnet50_recycling_retrained-2.pth ⁠).

•⁠  ⁠*Switching Models*:
  Use the ⁠ /switch_model ⁠ endpoint to switch between available models.

---

## Retraining the Model

1.⁠ ⁠Add new images to the ⁠ images/low_confidence/ ⁠ directory.

2.⁠ ⁠Use the ⁠ /retrain ⁠ endpoint to retrain the model.

3.⁠ ⁠The retrained model will be saved with a new version number.