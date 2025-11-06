import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from .schemas import PredictResponse, HealthResponse, ErrorResponse
from .inference import load, predict_image
from .db import log_prediction, get_recent
app = FastAPI(title="Image Classifier API", version="1.0.0")
@app.on_event("startup")
def _startup(): load()
@app.get("/health", response_model=HealthResponse)
def health():
    from .inference import _model, _classes
    return {"status":"ok","loaded":_model is not None,"classes":_classes or []}
@app.post("/predict", response_model=PredictResponse, responses={400: {"model": ErrorResponse}})
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Archivo no es imagen.")
    rid = str(uuid.uuid4())
    top1, prob, all_probs = predict_image(await file.read())
    log_prediction(rid, file.filename, top1, prob, all_probs)
    return {"filename": file.filename, "top1_class": top1, "top1_prob": prob, "topk": list(all_probs.keys()), "request_id": rid}
@app.get("/history")
def history(limit: int = 50): return get_recent(limit)
@app.get("/", response_class=HTMLResponse)
def index():
    try: return open("app/static/index.html","r",encoding="utf-8").read()
    except Exception: return "<h3>API lista. Visita /docs</h3>"
