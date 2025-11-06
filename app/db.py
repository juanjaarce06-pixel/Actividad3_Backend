import os
from google.cloud import firestore
_db = firestore.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
def log_prediction(request_id, filename, top1_class, top1_prob, classes_probs):
    _db.collection("predictions").document(request_id).set({
        "filename": filename,
        "top1_class": top1_class,
        "top1_prob": float(top1_prob),
        "probs": {k: float(v) for k, v in classes_probs.items()}
    })
def get_recent(limit=50):
    docs = (_db.collection("predictions").order_by("filename").limit(limit).stream())
    return [d.to_dict() for d in docs]
