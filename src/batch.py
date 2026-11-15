import json
import platform
from pathlib import Path
from importlib.metadata import version

import numpy as np


def source_identity(path):
    '''
    Identify a source by name, byte size and modification time.
    Reuse compares modification times at microsecond precision for portability.
    This detects common changes but is not a content checksum.
    '''
    path = Path(path)
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def pipeline_config(window_sec, stride_sec, num_frames, use_yolo):
    '''
    Describe the fixed encoder, sampling settings and relevant library versions.
    '''
    return {
        "python": platform.python_version(), "schema_version": 1, "encoder": "torchvision.r2plus1d_18",
        "weights": "R2Plus1D_18_Weights.KINETICS400_V1", "embedding_dim": 512,
        "preprocessing": "weights.transforms(); BGR to RGB",
        "frozen": True, "window_sec": window_sec, "stride_sec": stride_sec,
        "num_frames": num_frames, "use_yolo": use_yolo,
        "yolo_model": "yolo11n.pt" if use_yolo else None,
        "label_policy": "full_window_inside_segment; exclusive_end",
        "versions": {name: version(name) for name in
                     ("torch", "torchvision", "numpy", "opencv-python", "h5py")}
    }


def write_json(path, data):
    '''
    Write a new JSON file, refusing to replace an existing file.
    '''
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def load_completed(directory, source, config, camera):
    '''
    Reuse only completed, compatible records with valid referenced embeddings.
    Raise on partial outputs instead of overwriting or silently skipping them.
    '''
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        saved_source, current_source = manifest["source"].copy(), source.copy()
        # Copies between filesystems may round the final nanosecond digits.
        # Keep original timestamps in metadata; normalize only this comparison.
        for identity in (saved_source, current_source):
            identity["mtime_ns"] //= 1000
        if saved_source != current_source:
            changed = sorted(key for key in set(saved_source) | set(current_source)
                             if saved_source.get(key) != current_source.get(key))
            raise ValueError(f"Source changed: {', '.join(changed)}")
        if manifest["config"] != config:
            raise ValueError("Processing configuration changed")
        if manifest["camera"] != camera or not manifest["complete"]:
            raise ValueError("Camera changed or recording incomplete")
        clips = json.loads((directory / "clips.json").read_text())
        if not clips or len(clips) != manifest["clip_count"]:
            raise ValueError("Incomplete clip metadata")
        if len({c["clip_id"] for c in clips}) != len(clips):
            raise ValueError("Duplicate clip IDs")
        for clip in clips:
            embedding = np.load(directory / clip["embedding_path"], allow_pickle=False)
            if (embedding.shape != (config["embedding_dim"],)
                    or not np.isfinite(embedding).all() or np.linalg.norm(embedding) == 0):
                raise ValueError("Invalid cached embedding")
        return clips
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot reuse {directory}: {exc}. Use a new --output-dir.") from exc
