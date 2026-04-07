from pypdf import PdfReader
from .chunker import chunk_text

def process_pdf(path: str):
    r=PdfReader(path)
    full=[]
    for p in r.pages:
        t=p.extract_text() or ""
        if t.strip():
            full.append(t)
    text="\n".join(full)
    chunks=chunk_text(text)
    return [{"text":c,"t_start":0.0,"t_end":0.0,"ref_path":None,"source":"pdf"} for c in chunks]