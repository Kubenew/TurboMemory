import imagehash
from PIL import Image

def compute_phash(path: str):
    img = Image.open(path).convert("RGB")
    return str(imagehash.phash(img))

def hamming_distance(hash1: str, hash2: str):
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return int(h1 - h2)