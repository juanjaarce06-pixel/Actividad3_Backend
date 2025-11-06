import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from io import BytesIO
from .storage import download_model
_model, _classes = None, None
_pre = T.Compose([T.Resize((224,224)), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def load():
    global _model, _classes
    local = download_model()
    ckpt = torch.load(local, map_location="cpu")
    _classes = ckpt["classes"]
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(_classes))
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    _model = m; return True
def predict_image(file_bytes):
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    x = _pre(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=1)[0]
    top1 = int(probs.argmax().item())
    return _classes[top1], float(probs[top1]), { _classes[i]: float(probs[i]) for i in range(len(_classes)) }
