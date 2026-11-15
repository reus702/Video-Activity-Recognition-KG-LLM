from pathlib import Path

import h5py
import numpy as np


def extract_video(h5_path, output_dir, camera="hama1"):
    '''
    Save encoded video and original camera timestamps without replacing files.
    Return the MP4 path.
    '''
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{h5_path.stem}_{camera}"
    video_path = output_dir / f"{stem}.mp4"
    timestamp_path = output_dir / f"{stem}_timestamps.npy"
    if video_path.exists() or timestamp_path.exists():
        raise FileExistsError(f"Existing video or timestamps: {stem}")

    with h5py.File(h5_path, "r") as f:
        video_data = f[camera][()]
        timestamps = np.asarray(f[f"timestamps/{camera}"][()]).reshape(-1)

    # Exclusive creation also protects against accidental concurrent writes.
    with video_path.open("xb") as stream:
        stream.write(bytes(video_data))
    with timestamp_path.open("xb") as stream:
        np.save(stream, timestamps, allow_pickle=False)
    return video_path
