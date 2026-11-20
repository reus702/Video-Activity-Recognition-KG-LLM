import cv2
import h5py
import numpy as np


class OverlappingSegmentsError(ValueError):
    '''Raised when two annotation segments overlap.'''


def load_annotations(h5_path, camera="hama1"):
    '''
    Load and validate segment annotations and original camera timestamps.
    Reject missing fields, invalid time bounds and overlapping segments.
    '''
    annotations = []
    with h5py.File(h5_path, "r") as f:
        for key in (camera, f"timestamps/{camera}", "segments_info"):
            if key not in f:
                raise ValueError(f"Missing HDF5 field: {key}")
        timestamps = np.asarray(f[f"timestamps/{camera}"][()]).reshape(-1)
        if (len(timestamps) < 2 or not np.isfinite(timestamps).all()
                or not np.all(np.diff(timestamps) > 0)):
            raise ValueError("Camera timestamps must be finite and strictly increasing")

        for segment_id, segment in f["segments_info"].items():
            for key in ("text", "start", "end", "success"):
                if key not in segment:
                    raise ValueError(f"Segment {segment_id}: missing {key}")
            raw_text = segment["text"][()]
            text = (raw_text.decode() if isinstance(raw_text, bytes) else str(raw_text))
            text = text.strip().rstrip(".").strip()
            if not text:
                raise ValueError(f"Segment {segment_id}: empty label")
            if text.lower() == "no action":
                action, obj = "IDLE", None
            else:
                parts = text.split(maxsplit=1)
                action, obj = parts[0].upper(), parts[1] if len(parts) > 1 else None
            start, end = float(segment["start"][()]), float(segment["end"][()])
            if not np.isfinite([start, end]).all() or start >= end:
                raise ValueError(f"Segment {segment_id}: invalid time bounds")
            success = segment["success"][()]
            if np.ndim(success) != 0 or success not in (True, False):
                raise ValueError(f"Segment {segment_id}: invalid success flag")
            annotations.append({
                "segment_id": segment_id, "start": start, "end": end,
                "action": action, "object": obj, "success": bool(success)
            })
    if not annotations:
        raise ValueError("No segment annotations found")
    annotations.sort(key=lambda ann: ann["start"])
    for previous, current in zip(annotations, annotations[1:]):
        if current["start"] < previous["end"]:
            raise OverlappingSegmentsError("Overlapping annotation segments")
    return annotations, timestamps


def validate_video(video_path, timestamps):
    '''
    Check video frame count against camera timestamps and return FPS and count.
    This structural check does not prove temporal synchronization.
    '''
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid video FPS")
    if n_frames != len(timestamps):
        raise ValueError("Video frame count differs from timestamp count")
    return fps, n_frames


def label_clips(h5_path, video_path, clips, camera="hama1"):
    '''
    Label windows fully inside one segment and retain original time bounds.
    Return metadata copies; unmatched windows keep a None label.
    '''
    annotations, timestamps = load_annotations(h5_path, camera)
    fps, n_frames = validate_video(video_path, timestamps)
    labeled_clips = []
    for original in clips:
        clip = original.copy()
        # Fall back to relative times for metadata from older notebooks.
        start_idx = clip.get("start_frame", round(clip["start_time"] * fps))
        end_idx = clip.get("end_frame_exclusive", round(clip["end_time"] * fps))
        if not 0 <= start_idx < end_idx <= n_frames:
            raise ValueError("Clip frame bounds fall outside the video")
        start = float(timestamps[start_idx])
        # Preserve the existing exclusive-end labeling convention.
        end = float(timestamps[end_idx] if end_idx < n_frames
                    else timestamps[-1] + np.median(np.diff(timestamps)))
        clip.update({
            "start_frame": start_idx, "end_frame_exclusive": end_idx,
            "original_start_time": start, "original_end_time_exclusive": end,
            "end_time_estimated": end_idx == n_frames,
            "ground_truth": None, "object": None, "success": None, "segment_id": None
        })
        if "sampled_frame_indices" in clip:
            indices = np.asarray(clip["sampled_frame_indices"])
            if (not np.issubdtype(indices.dtype, np.integer)
                    or np.any(indices < start_idx) or np.any(indices >= end_idx)):
                raise ValueError("Sampled frames fall outside the clip")
            clip["original_frame_times"] = timestamps[indices].tolist()
        for ann in annotations:
            if start >= ann["start"] and end <= ann["end"]:
                clip.update({"ground_truth": ann["action"], "object": ann["object"],
                             "success": ann["success"], "segment_id": ann["segment_id"]})
                break
        labeled_clips.append(clip)
    return labeled_clips
