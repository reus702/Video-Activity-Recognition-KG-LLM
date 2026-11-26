import cv2
import numpy as np


def read_sampled_frames(video_path, indices):
    '''Read saved frame indices in order with a single video seek.'''
    indices = tuple(indices)
    if not indices or indices[0] < 0 or any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("Expected increasing, nonnegative frame indices")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0]):
            raise ValueError(f"Cannot seek to frame {indices[0]} in {video_path}")
        frames = []
        for index in range(indices[0], indices[-1] + 1):
            if not capture.grab():
                raise ValueError(f"Cannot read frame {index} in {video_path}")
            if index == indices[len(frames)]:
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError(f"Cannot decode frame {index} in {video_path}")
                frames.append(frame)
        return frames
    finally:
        capture.release()


def iter_clip_metadata(fps, total, window_sec=3.0, stride_sec=1.5, num_frames=16):
    """Describe complete sliding windows without decoding video frames."""
    if (not np.isfinite([window_sec, stride_sec]).all()
            or window_sec <= 0 or stride_sec <= 0
            or not isinstance(num_frames, int) or num_frames < 1):
        raise ValueError("Invalid clip sampling parameters")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS non validi")
    window = round(window_sec * fps)
    stride = round(stride_sec * fps)
    if window < num_frames or stride < 1:
        raise ValueError("Parametri non validi")
    offsets = np.linspace(0, window - 1, num_frames, dtype=int)
    for start in range(0, total - window + 1, stride):
        indices = (start + offsets).tolist()
        yield {
            "start_frame": start,
            "end_frame_exclusive": start + window,
            "sampled_frame_indices": indices,
            "start_time": start / fps,
            "end_time": (start + window) / fps,
            "frame_times": [index / fps for index in indices],
        }


def iter_clips(video_path, window_sec=3.0, stride_sec=1.5, num_frames=16):
    """Yield sampled BGR frames from sliding windows with relative video times."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError("Impossibile aprire il video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for clip in iter_clip_metadata(fps, total, window_sec, stride_sec, num_frames):
            start = clip["start_frame"]
            indices = set(clip["sampled_frame_indices"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            for index in range(start, clip["end_frame_exclusive"]):
                ok, frame = cap.read()
                if not ok:
                    break
                if index in indices:
                    frames.append(frame)
            if len(frames) == num_frames:
                yield {**clip, "frames": frames}
    finally:
        cap.release()
