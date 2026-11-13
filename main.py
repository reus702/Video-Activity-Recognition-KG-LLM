import json
from pathlib import Path
from collections import Counter

import numpy as np

from src.dataset import extract_video
from src.video_processing import iter_clips
from src.feature_extraction import VideoFeatureExtractor
from src.object_detection import ObjectDetector
from src.labeling import label_clips


ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "data/raw/reassemble"
VIDEO_DIR = ROOT / "data/videos"
OUTPUT_DIR = ROOT / "outputs"
EMBEDDING_DIR = OUTPUT_DIR / "embeddings"

CAMERA = "hama1"
USE_YOLO = False


def main():

    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cerca tutti i file HDF5
    h5_files = sorted(RAW_DIR.rglob("*.h5"))

    print(f"Trovati {len(h5_files)} file HDF5")

    if not h5_files:
        raise FileNotFoundError("Nessun file HDF5 trovato")

    # 2. Carica i modelli una sola volta
    extractor = VideoFeatureExtractor()

    detector = ObjectDetector() if USE_YOLO else None

    # 3. Elabora ogni registrazione
    for h5_path in h5_files:

        print(f"\nProcessing: {h5_path.name}")

        # Estrazione MP4
        video_path = extract_video(
            h5_path,
            VIDEO_DIR,
            camera=CAMERA
        )

        video_id = video_path.stem
        metadata = []

        # 4. Creazione delle clip
        for i, clip in enumerate(iter_clips(video_path)):

            clip_id = f"{video_id}_clip_{i:04d}"

            # Estrazione embedding
            embedding = extractor.extract(clip["frames"])

            # Object detection (opzionale)
            detections = []

            if detector is not None:

                for j in [0, 7, 15]:

                    detections.append({
                        "timestamp": clip["frame_times"][j],
                        "objects": detector.detect(
                            clip["frames"][j]
                        )
                    })

            # Salvataggio embedding
            embedding_path = EMBEDDING_DIR / f"{clip_id}.npy"

            np.save(embedding_path, embedding)

            metadata.append({
                "clip_id": clip_id,
                "video_id": video_id,
                "recording_id": h5_path.stem,
                "start_time": clip["start_time"],
                "end_time": clip["end_time"],
                "embedding_path": str(
                    embedding_path.relative_to(ROOT)
                ),
                "detections": detections
            })

            print(clip_id, embedding.shape)

        # 5. Associazione ground truth
        labeled_clips = label_clips(
            h5_path,
            video_path,
            metadata,
            camera=CAMERA
        )

        # 6. Salvataggio JSON per ogni video
        output_path = OUTPUT_DIR / f"{video_id}_labeled.json"

        with open(output_path, "w") as f:
            json.dump(labeled_clips, f, indent=2)

        # 7. Riepilogo azioni trovate
        counts = Counter(
            c["ground_truth"] for c in labeled_clips
        )

        print("Ground truth:", counts)
        print("Saved:", output_path)

if __name__ == "__main__":
    main()