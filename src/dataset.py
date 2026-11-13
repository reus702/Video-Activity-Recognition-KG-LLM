from pathlib import Path

import h5py
import numpy as np
'''
Function used to save a video and to extract camera timestamps from HDF5 in a preliminary phase
'''
def extract_video(h5_path, output_dir, camera="hama1"):

    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{h5_path.stem}_{camera}"
    video_path = output_dir / f"{stem}.mp4"

    'Extracting video from .h5 file to .mp4'
    with h5py.File(h5_path, "r") as f:

        video_data = f[camera][()]

        video_path.write_bytes(bytes(video_data))

        timestamps = np.asarray(
            f[f"timestamps/{camera}"][()]
        ).reshape(-1)

        np.save(
            output_dir / f"{stem}_timestamps.npy",
            timestamps
        )

    return video_path