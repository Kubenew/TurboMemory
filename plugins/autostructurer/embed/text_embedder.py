import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", use_gpu=True):
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts):
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return vecs.astype(np.float32)