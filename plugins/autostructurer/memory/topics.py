import numpy as np

def assign_topic(vec: np.ndarray, centroids: np.ndarray):
    if centroids is None or len(centroids)==0:
        return 0, vec.copy()
    sims = centroids @ vec
    idx = int(np.argmax(sims))
    return idx, centroids[idx]

def update_centroid(old: np.ndarray, new: np.ndarray, alpha=0.05):
    return (1-alpha)*old + alpha*new