import numpy as np
from .pack4bit import unpack_4bit

def unpack_vectors_for_search(rows):
    vecs=[]
    for r in rows:
        dim=r[2]; packed=r[3]; scale=r[4]; zero=r[5]
        v=unpack_4bit(packed, dim, scale, zero)
        vecs.append(v)
    return np.vstack(vecs).astype(np.float32)