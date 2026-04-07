import cv2
import os

def extract_scene_keyframes(video_path: str, out_dir: str, threshold=30.0, step=10):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    frames = []
    last_gray = None
    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step != 0:
            idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if last_gray is None:
            last_gray = gray
            idx += 1
            continue

        diff = cv2.absdiff(gray, last_gray)
        score = float(diff.mean())

        if score > threshold:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            out_path = os.path.join(out_dir, f"scene_{saved:06d}.jpg")
            cv2.imwrite(out_path, frame)
            frames.append((t, out_path, score))
            saved += 1
            last_gray = gray

        idx += 1

    cap.release()
    return frames