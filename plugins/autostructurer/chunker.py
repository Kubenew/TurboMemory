def chunk_text(text: str, max_chars=1200, overlap=200):
    text = text.strip()
    if not text:
        return []
    chunks=[]
    i=0
    while i < len(text):
        end=min(len(text), i+max_chars)
        c=text[i:end].strip()
        if c:
            chunks.append(c)
        i=end-overlap
        if i < 0:
            i=0
        if end==len(text):
            break
    return chunks