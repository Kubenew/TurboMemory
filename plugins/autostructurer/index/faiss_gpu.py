import faiss
import numpy as np

def make_gpu_resources():
    res = faiss.StandardGpuResources()
    return res

def to_gpu(index_cpu):
    res = make_gpu_resources()
    return faiss.index_cpu_to_gpu(res, 0, index_cpu)

def to_cpu(index_gpu):
    return faiss.index_gpu_to_cpu(index_gpu)