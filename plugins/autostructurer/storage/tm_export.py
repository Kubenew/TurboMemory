import os, json, struct, sqlite3, shutil
import faiss

def export_tm(db_path: str, text_index_path: str, clip_index_path: str, out_tm: str):
    tmp_dir = out_tm + "_dir"
    os.makedirs(tmp_dir, exist_ok=True)

    shutil.copy(db_path, os.path.join(tmp_dir, "memory.sqlite"))

    if os.path.exists(text_index_path):
        shutil.copy(text_index_path, os.path.join(tmp_dir, "faiss_text.index"))
    if os.path.exists(clip_index_path):
        shutil.copy(clip_index_path, os.path.join(tmp_dir, "faiss_clip.index"))

    header = {
        "version": 1,
        "files": {
            "sqlite": "memory.sqlite",
            "faiss_text": "faiss_text.index",
            "faiss_clip": "faiss_clip.index"
        }
    }

    header_json = json.dumps(header).encode("utf-8")

    with open(out_tm, "wb") as f:
        f.write(b"TMEMv1\n")
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)

    return tmp_dir