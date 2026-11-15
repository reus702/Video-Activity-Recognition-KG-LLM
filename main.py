import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.batch import load_completed, pipeline_config, source_identity, write_json
from src.dataset import extract_video
from src.dataset_report import summarize_dataset
from src.labeling import label_clips, load_annotations, validate_video
from src.video_processing import iter_clips

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data/raw/reassemble"
OUTPUT_DIR = ROOT / "outputs/visual_baseline"
CAMERA = "hama1"
USE_YOLO = False
WINDOW_SEC = 3.0
STRIDE_SEC = 1.5
NUM_FRAMES = 16


def main():
    '''
    Process new recordings and reuse verified completed outputs.
    Keep legacy outputs untouched and stop on incomplete or incompatible records.
    '''
    parser = argparse.ArgumentParser(description="Prepare the visual baseline dataset")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    raw_dir, output_dir = args.raw_dir.resolve(), args.output_dir.resolve()
    h5_files = sorted(raw_dir.rglob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 recordings found in {raw_dir}")

    # Check all IDs before creating any recording outputs.
    ids = [path.stem for path in h5_files]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate recording names in input folders")
    config = pipeline_config(WINDOW_SEC, STRIDE_SEC, NUM_FRAMES, USE_YOLO)
    extractor = detector = None
    all_clips = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for h5_path in h5_files:
        source = source_identity(h5_path)
        source["relative_path"] = str(h5_path.relative_to(raw_dir))
        video_id = f"{h5_path.stem}_{CAMERA}"
        directory = output_dir / video_id
        if directory.exists():
            all_clips.extend(load_completed(directory, source, config, CAMERA))
            print(f"Reused: {video_id}")
            continue

        # Validate annotations before extracting video or loading model weights.
        _, timestamps = load_annotations(h5_path, CAMERA)
        directory.mkdir()  # Claim this recording without replacing existing files.
        video_path = extract_video(h5_path, directory, CAMERA)
        fps, n_frames = validate_video(video_path, timestamps)
        window, stride = round(WINDOW_SEC * fps), round(STRIDE_SEC * fps)
        expected_count = max(0, (n_frames - window) // stride + 1)
        if expected_count == 0:
            raise ValueError(f"Recording too short: {h5_path.name}")
        if extractor is None:
            from src.feature_extraction import VideoFeatureExtractor
            extractor = VideoFeatureExtractor()
            if USE_YOLO:
                from src.object_detection import ObjectDetector
                detector = ObjectDetector()
        (directory / "embeddings").mkdir()
        metadata = []
        for i, clip in enumerate(iter_clips(video_path, WINDOW_SEC, STRIDE_SEC, NUM_FRAMES)):
            clip_id = f"{video_id}_clip_{i:04d}"
            embedding = extractor.extract(clip["frames"])
            if (embedding.shape != (512,) or not np.isfinite(embedding).all()
                    or np.linalg.norm(embedding) == 0):
                raise ValueError(f"Invalid embedding: {clip_id}")
            embedding_path = Path("embeddings") / f"{clip_id}.npy"
            with (directory / embedding_path).open("xb") as stream:
                np.save(stream, embedding, allow_pickle=False)
            detections = []
            if detector is not None:
                for j in (0, 7, 15):
                    detections.append({"timestamp": clip["frame_times"][j],
                                       "objects": detector.detect(clip["frames"][j])})
            metadata.append({
                **{key: value for key, value in clip.items() if key != "frames"},
                "clip_id": clip_id, "video_id": video_id,
                "recording_id": h5_path.stem, "camera": CAMERA,
                "embedding_path": str(embedding_path), "detections": detections
            })
            print(clip_id)
        if len(metadata) != expected_count:
            raise ValueError("Some expected windows could not be decoded")
        labeled = label_clips(h5_path, video_path, metadata, CAMERA)
        if source_identity(h5_path) != {k: v for k, v in source.items() if k != "relative_path"}:
            raise ValueError("Source changed during processing; wait for download completion")
        write_json(directory / "clips.json", labeled)
        # Write completion last, so interrupted runs cannot be mistaken for complete runs.
        write_json(directory / "manifest.json", {
            "complete": True, "recording_id": h5_path.stem, "camera": CAMERA,
            "source": source, "config": config, "clip_count": len(labeled),
            "video_fps": fps, "video_frame_count": n_frames,
            "embedding_paths_relative_to": "recording directory"
        })
        all_clips.extend(labeled)
        print(f"Completed: {video_id}")

    # Each report is a new snapshot; earlier reports remain available.
    report = summarize_dataset(all_clips)
    report["input_root"] = str(raw_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = output_dir / f"dataset_report_{stamp}.json"
    write_json(report_path, report)
    print(f"Report: {report_path} ({report['recording_count']} recordings, {report['clip_count']} clips)")


if __name__ == "__main__":
    main()
