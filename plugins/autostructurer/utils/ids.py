import hashlib
def sha1_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]