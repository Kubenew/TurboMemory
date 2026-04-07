import easyocr
_reader=None

def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en","ru","cs"], gpu=True)
    return _reader

def process_image(path: str):
    reader=get_reader()
    lines=reader.readtext(path, detail=0)
    text=" ".join([x.strip() for x in lines if x.strip()]).strip()
    if not text:
        return []
    return [{"text":text,"t_start":0.0,"t_end":0.0,"ref_path":path,"source":"ocr"}]