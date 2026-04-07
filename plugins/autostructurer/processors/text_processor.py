from .chunker import chunk_text

def process_text_file(path: str):
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        text=f.read()
    chunks=chunk_text(text)
    return [{"text":c,"t_start":0.0,"t_end":0.0,"ref_path":None,"source":"text"} for c in chunks]