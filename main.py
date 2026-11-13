import json
from pathlib import Path

import numpy as np

from src.video_processing import iter_clips
from src.feature_extraction import VideoFeatureExtractor
from src.object_detection import ObjectDetector


def main():

    video_path = Path("data/videos/2025-01-09-13-57-17_hama1.mp4")

    output_dir = Path("outputs")
    embedding_dir = output_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    extractor = VideoFeatureExtractor()
    detector = ObjectDetector()

    metadata = []

    for i, clip in enumerate(iter_clips(video_path)):

        clip_id = f"{video_path.stem}_clip_{i:04d}"

        # 1. Feature extraction
        embedding = extractor.extract(clip["frames"])

        # 2. Object detection
        detections = []

        for j in [0, 7, 15]:

            detections.append({
                "timestamp": clip["frame_times"][j],
                "objects": detector.detect(
                    clip["frames"][j]
                )
            })

        # 3. Salvataggio embedding
        embedding_path = embedding_dir / f"{clip_id}.npy"

        np.save(embedding_path, embedding)

        # 4. Salvataggio metadati
        metadata.append({
            "clip_id": clip_id,
            "video_id": video_path.stem,
            "start_time": clip["start_time"],
            "end_time": clip["end_time"],
            "embedding_path": str(embedding_path),
            "detections": detections
        })

        print(
            clip_id,
            "Embedding shape:", embedding.shape
        )

    with open(output_dir / "clips.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()