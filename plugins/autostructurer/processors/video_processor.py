import os
import ffmpeg
import whisper
import easyocr
import cv2
from tqdm import tqdm

def extract_audio(video_path: str, out_wav: str):
    ffmpeg.input(video_path).output(out_wav, ac=1, ar=16000).overwrite_output().run(quiet=True)

def extract_scene_keyframes(video_path: str, out_dir: str, threshold=30.0, step=10):
    os.makedirs(out_dir, exist_ok=True)
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    frames=[]
    last=None
    idx=0
    saved=0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step != 0:
            idx += 1
            continue

        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if last is None:
            last=gray
            idx += 1
            continue

        diff=cv2.absdiff(gray,last)
        score=float(diff.mean())

        if score > threshold:
            t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
            fp=os.path.join(out_dir, f"scene_{saved:06d}.jpg")
            cv2.imwrite(fp, frame)
            frames.append((t, fp))
            saved += 1
            last=gray

        idx += 1

    cap.release()
    return frames

def process_video(path: str, temp_dir="tmp_video", scene_threshold=30.0):
    os.makedirs(temp_dir, exist_ok=True)
    audio=os.path.join(temp_dir, "audio.wav")
    extract_audio(path, audio)

    w=whisper.load_model("base")
    res=w.transcribe(audio)

    chunks=[]
    for seg in res["segments"]:
        chunks.append({
            "text": seg["text"].strip(),
            "t_start": float(seg["start"]),
            "t_end": float(seg["end"]),
            "ref_path": None,
            "source": "speech"
        })

    scenes_dir=os.path.join(temp_dir, "scenes")
    scenes=extract_scene_keyframes(path, scenes_dir, threshold=scene_threshold, step=10)

    reader=easyocr.Reader(["en","ru","cs"], gpu=True)
    for t, fp in tqdm(scenes, desc="OCR scenes"):
        lines=reader.readtext(fp, detail=0)
        text=" ".join([x.strip() for x in lines if x.strip()]).strip()
        if not text:
            continue
        chunks.append({
            "text": text,
            "t_start": t,
            "t_end": t+2.0,
            "ref_path": fp,
            "source": "ocr_scene"
        })

    return chunks