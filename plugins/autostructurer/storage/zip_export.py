import shutil, os

def export_zip(export_dir: str, zip_path: str):
    base = zip_path
    if base.endswith(".zip"):
        base = base[:-4]
    shutil.make_archive(base, "zip", export_dir)
    return base + ".zip"