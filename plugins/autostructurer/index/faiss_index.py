import os
import numpy as np
import faiss
from ..config import Config
from .faiss_gpu import to_gpu, to_cpu

class IVFIndex:
    def __init__(self, dim: int, path: str, use_gpu=True):
        self.dim = dim
        self.path = path
        self.use_gpu = use_gpu
        self.index = None
        self.trained = False

        self._load_or_create()

    def _create_cpu(self):
        quant = faiss.IndexFlatIP(self.dim)
        idx = faiss.IndexIVFPQ(quant, self.dim, Config.IVF_NLIST, Config.IVF_M, Config.IVF_NBITS)
        idx.nprobe = 16
        return idx

    def _load_or_create(self):
        if os.path.exists(self.path):
            idx = faiss.read_index(self.path)
        else:
            idx = self._create_cpu()
        self.trained = idx.is_trained
        self.index = to_gpu(idx) if self.use_gpu else idx

    def save(self):
        cpu = to_cpu(self.index) if self.use_gpu else self.index
        faiss.write_index(cpu, self.path)

    def train_if_needed(self, vectors: np.ndarray):
        if self.trained:
            return
        if vectors.shape[0] < Config.TRAIN_MIN_VECTORS:
            return
        cpu = to_cpu(self.index) if self.use_gpu else self.index
        cpu.train(vectors)
        self.trained = True
        self.index = to_gpu(cpu) if self.use_gpu else cpu

    def add(self, vectors: np.ndarray, ids: np.ndarray):
        self.train_if_needed(vectors)
        self.index.add_with_ids(vectors, ids)

    def search(self, qvecs: np.ndarray, top_k=10):
        return self.index.search(qvecs, top_k)