import os
from google.cloud import storage
BUCKET = os.getenv("MODEL_BUCKET")
OBJECT = os.getenv("MODEL_PATH", "model/model.pt")
def download_model(local_path="/tmp/model.pt"):
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    bucket.blob(OBJECT).download_to_filename(local_path)
    return local_path
