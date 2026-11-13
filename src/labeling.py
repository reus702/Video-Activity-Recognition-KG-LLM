import cv2
import h5py
import numpy as np


def load_annotations(h5_path, camera="hama1"):
    annotations = []

    with h5py.File(h5_path, "r") as f:

        timestamps = np.asarray(
            f[f"timestamps/{camera}"][()]
        ).reshape(-1)

        for segment_id, segment in f["segments_info"].items():

            text = segment["text"][()].decode().strip(".")

            if text.lower() == "no action":
                action, obj = "IDLE", None
            else:
                action, _, obj = text.partition(" ")

            annotations.append({
                "segment_id": segment_id,
                "start": float(segment["start"][()]),
                "end": float(segment["end"][()]),
                "action": action.upper(),
                "object": obj,
                "success": bool(segment["success"][()])
            })

    return annotations, timestamps


def label_clips(h5_path, video_path, clips, camera="hama1"):

    annotations, timestamps = load_annotations(
        h5_path, camera
    )

    # Recupera informazioni sul video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Verifica sincronizzazione
    if fps <= 0:
        raise ValueError("FPS non validi")

    if n_frames != len(timestamps):
        raise ValueError("Frame e timestamp non corrispondono")

    if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0):
        raise ValueError("Timestamp non validi")

    labeled_clips = []

    for clip in clips:

        # Copia i metadati della clip
        clip = clip.copy()

        # Converte i tempi relativi in indici dei frame
        start_idx = round(clip["start_time"] * fps)
        end_idx = round(clip["end_time"] * fps)

        # Recupera i timestamp originali
        start = timestamps[start_idx]

        end = (
            timestamps[end_idx]
            if end_idx < len(timestamps)
            else timestamps[-1] + np.median(np.diff(timestamps))
        )

        # Valori di default
        clip["ground_truth"] = None
        clip["object"] = None
        clip["success"] = None
        clip["segment_id"] = None

        # Cerca il segmento contenente l'intera clip
        for ann in annotations:

            if start >= ann["start"] and end <= ann["end"]:

                clip["ground_truth"] = ann["action"]
                clip["object"] = ann["object"]
                clip["success"] = ann["success"]
                clip["segment_id"] = ann["segment_id"]

                break

        labeled_clips.append(clip)

    return labeled_clips