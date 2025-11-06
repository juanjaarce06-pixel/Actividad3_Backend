# app/inference.py (MOCK - sin PyTorch)
from PIL import Image
from io import BytesIO
import hashlib

# Clases de ejemplo
_classes = ["cat", "dog", "bird"]

def load():
    # No carga de modelo; deja todo listo para /health
    return True

def _stable_probs_from_bytes(b: bytes, n: int):
    # Genera probabilidades pseudo-determinísticas a partir del hash del archivo
    h = hashlib.sha256(b).digest()
    vals = [h[i] / 255 for i in range(n)]
    s = sum(vals) or 1.0
    return [v / s for v in vals]

def predict_image(file_bytes):
    # Abrir imagen (valida formato) — no se usa para inferencia real
    Image.open(BytesIO(file_bytes)).convert("RGB")

    probs = _stable_probs_from_bytes(file_bytes, len(_classes))
    # Top-1
    top1_idx = max(range(len(probs)), key=lambda i: probs[i])
    top1 = _classes[top1_idx]
    # Dict de probabilidades
    all_probs = { _classes[i]: float(probs[i]) for i in range(len(_classes)) }
    return top1, float(probs[top1_idx]), all_probs
