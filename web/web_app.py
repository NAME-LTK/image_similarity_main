from base64 import b64encode
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms as T

from image_classification import classification_config as cls_config
from image_classification.classification_model import Classifier
from image_denoising import denoising_config as config
from image_denoising.denoising_model import ConvDenoiser
from image_similarity import similarity_config as sim_config
from image_similarity.similarity_search import ImageSearcher

app = FastAPI()
app.mount(
    "/dataset",
    StaticFiles(directory=str(sim_config.IMG_PATH)),
    name="dataset",
)
searcher = ImageSearcher()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denoiser = ConvDenoiser().to(device)
denoiser.load_state_dict(
    torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
)
denoiser.eval()

transform = T.Compose(
    [
        T.Resize(config.IMG_SIZE),
        T.ToTensor(),
    ]
)
to_pil = T.ToPILImage()

classifier = Classifier(num_classes=len(cls_config.CLASS_NAMES)).to(device)
classifier.load_state_dict(
    torch.load(cls_config.MODEL_PATH, map_location=device, weights_only=True)
)
classifier.eval()

classification_transform = T.Compose(
    [
        T.Resize(cls_config.IMG_SIZE),
        T.ToTensor(),
    ]
)


def encode_png(pixels):
    image = to_pil(pixels.cpu())
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return b64encode(buffer.getvalue()).decode("ascii")


@app.post("/denoising")
def denoising(image: UploadFile):
    with Image.open(image.file) as uploaded:
        clean = transform(uploaded.convert("RGB"))

    noise = torch.randn_like(clean) * config.NOISE_FACTOR
    noisy = (clean + noise).clamp(0, 1)

    with torch.no_grad():
        denoised = denoiser(noisy.unsqueeze(0).to(device)).squeeze(0)

    return {
        "noisy_img": encode_png(noisy),
        "denoised_image": encode_png(denoised),
    }


@app.post("/classification", response_class=PlainTextResponse)
def classification(image: UploadFile):
    with Image.open(image.file) as uploaded:
        pixels = classification_transform(uploaded.convert("RGB"))

    batch = pixels.unsqueeze(0).to(device)
    with torch.no_grad():
        scores = classifier(batch)

    class_id = scores.argmax(dim=1).item()
    return f"您搜索的商品类型是：{cls_config.CLASS_NAMES[class_id]}"


@app.post("/simimages")
def simimages(image: UploadFile):
    with Image.open(image.file) as uploaded:
        results = searcher.search(uploaded, top_k=5)

    return {"image_urls": [f"/dataset/{image_path.name}" for image_path, _ in results]}


@app.get("/")
def index():
    page_path = Path(__file__).resolve().with_name("index.html")
    return FileResponse(page_path, media_type="text/html")
