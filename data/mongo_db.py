from pymongo import MongoClient
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv

# Load Configuration
load_dotenv(dotenv_path="./.env.local")

# Load env variables
DB_URI = os.environ.get("DB_URI", "")
DB_NAME = os.environ.get("DB_NAME", "")
SAMPLE_COLLECTION = os.environ.get("SAMPLE_COLLECTION", "")

# Connect to MongoDB
client = MongoClient(DB_URI)
db = client[DB_NAME]
samples_collection = db[SAMPLE_COLLECTION]
print("MongoDB connection alive, DB Collections: ", db.list_collection_names())

# Allowed values for fields
ALLOWED_SYSTEM_ANALYSIS = {"Paper", "Plastic", "Other", "Uncertain"}
ALLOWED_IMAGE_CLASS = {"Paper", "Plastic", "Other", None}
ALLOWED_OUTCOME = {"Success", "Failure", None}


def validate_sample(sample_data, require_all_fields=True):
    """
    Validates Sample data according to the specified constraints.
    
    :param sample_data: dict containing fields: ImageName, FilePath, SystemAnalysis, ImageClass, Outcome
    :param require_all_fields: bool indicating whether all fields must be present (for creation).
                               If False, only validate the fields that are present (for partial updates).
    :raises ValueError: if validation fails.
    """
    required_fields = ["ImageName", "FilePath", "SystemAnalysis", "ImageClass", "Outcome"]
    
    # If we require all fields (create scenario), ensure none are missing
    if require_all_fields:
        for field in required_fields:
            if field not in sample_data:
                raise ValueError(f"Missing required field '{field}'.")

    # Validate each field that is present
    if "ImageName" in sample_data:
        if not isinstance(sample_data["ImageName"], str):
            raise ValueError("ImageName must be a string.")

    if "FilePath" in sample_data:
        if not isinstance(sample_data["FilePath"], str):
            raise ValueError("FilePath must be a string.")

    if "SystemAnalysis" in sample_data:
        if sample_data["SystemAnalysis"] not in ALLOWED_SYSTEM_ANALYSIS:
            raise ValueError(f"SystemAnalysis must be one of {ALLOWED_SYSTEM_ANALYSIS}.")

    if "ImageClass" in sample_data:
        if sample_data["ImageClass"] not in ALLOWED_IMAGE_CLASS:
            raise ValueError(f"ImageClass must be one of {ALLOWED_IMAGE_CLASS}.")

    if "Outcome" in sample_data:
        if sample_data["Outcome"] not in ALLOWED_OUTCOME:
            raise ValueError(f"Outcome must be one of {ALLOWED_OUTCOME}.")


def create_sample(sample_data):
    """
    Creates a new Sample document in MongoDB.
    :param sample_data: dict containing all Sample fields
    :return: inserted_id (string)
    """
    # Validate input
    validate_sample(sample_data, require_all_fields=True)

    result = samples_collection.insert_one(sample_data)
    return str(result.inserted_id)


def get_samples():
    """
    Returns all Sample documents.
    :return: list of Sample dicts
    """
    docs = samples_collection.find()
    samples = []
    for doc in docs:
        doc["_id"] = str(doc["_id"])
        samples.append(doc)
    return samples


def get_sample_by_id(sample_id):
    """
    Returns a single Sample by _id.
    :param sample_id: string (ObjectId as string)
    :return: Sample dict or None if not found
    """
    doc = samples_collection.find_one({"_id": ObjectId(sample_id)})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


def get_sample_by_name(name):
    """
    Returns a single Sample by ImageName.
    :param name: string
    :return: Sample dict or None if not found
    """
    doc = samples_collection.find_one({"ImageName": name})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


def update_sample(sample_id, update_data):
    """
    Updates an existing Sample document by _id.
    :param sample_id: string (ObjectId as string)
    :param update_data: dict of fields to update
    :return: number of modified documents (int)
    """
    # Validate updated fields (partial validation)
    validate_sample(update_data, require_all_fields=False)

    result = samples_collection.update_one(
        {"_id": ObjectId(sample_id)},
        {"$set": update_data}
    )
    return result.modified_count


def delete_sample(sample_id):
    """
    Deletes a Sample document by _id.
    :param sample_id: string (ObjectId as string)
    :return: number of deleted documents (int)
    """
    result = samples_collection.delete_one({"_id": ObjectId(sample_id)})
    return result.deleted_count
