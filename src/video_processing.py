import cv2
import numpy as np

'''
Function used to process a video in temporal windows, windows containes multiple frames.
'''
def iter_clips( video_path, window_sec=3.0, stride_sec=1.5, num_frames=16 ):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Impossibile aprire il video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        if fps <= 0:
            raise ValueError("FPS non validi")

        window = round(window_sec * fps)
        stride = round(stride_sec * fps)

        if window < num_frames or stride < 1:
            raise ValueError("Parametri non validi")

        for start in range(0, total - window + 1, stride):

            indices = set(
                np.linspace(
                    0, window - 1, num_frames,
                    dtype=int
                )
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            frames = []
            timestamps = []

            for j in range(window):
                ok, frame = cap.read()

                if not ok:
                    break

                if j in indices:
                    frames.append(frame)
                    timestamps.append((start + j) / fps)

            if len(frames) == num_frames:
                yield {
                    "start_time": start / fps,
                    "end_time": (start + window) / fps,
                    "frame_times": timestamps,
                    "frames": frames
                }

    finally:
        cap.release()