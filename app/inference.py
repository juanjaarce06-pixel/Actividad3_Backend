cat > app/inference.py <<'EOF'
from PIL import Image
from io import BytesIO
import hashlib

# Variables globales esperadas por /health
_model = True  # cualquier valor "truthy" indica que el modelo está cargado
_classes = ["cat", "dog", "bird"]

def load():
    # Para el mock no hay nada que cargar, pero mantenemos la firma
    global _model
    _model = True
    return True

def _stable_probs_from_bytes(b: bytes, n: int):
    # Probabilidades pseudo-determinísticas a partir del hash del archivo
    h = hashlib.sha256(b).digest()
    vals = [h[i] / 255 for i in range(n)]
    s = sum(vals) or 1.0
    return [v / s for v in vals]

def predict_image(file_bytes):
    # Validar que es imagen
    Image.open(BytesIO(file_bytes)).convert("RGB")

    probs = _stable_probs_from_bytes(file_bytes, len(_classes))
    top1_idx = max(range(len(probs)), key=lambda i: probs[i])
    top1 = _classes[top1_idx]
    all_probs = { _classes[i]: float(probs[i]) for i in range(len(_classes)) }
    return top1, float(probs[top1_idx]), all_probs
EOF
