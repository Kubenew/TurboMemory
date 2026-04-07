import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_images(self, image_paths):
        images=[Image.open(p).convert("RGB") for p in image_paths]
        inputs=self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats=self.model.get_image_features(**inputs)
        vecs=feats.detach().cpu().numpy().astype(np.float32)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        return vecs

    def embed_texts(self, texts):
        inputs=self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feats=self.model.get_text_features(**inputs)
        vecs=feats.detach().cpu().numpy().astype(np.float32)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        return vecs