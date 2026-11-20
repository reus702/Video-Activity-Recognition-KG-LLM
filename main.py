import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.batch import load_completed, pipeline_config, source_identity, write_json
from src.dataset import extract_video
from src.dataset_report import summarize_dataset
from src.labeling import label_clips, load_annotations, validate_video
from src.reassemble import load_reassemble_plan, prepare_reassemble, recording_paths
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
    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command == "prepare-reassemble":
        prepare_reassemble_main()
        return
    if command in ("prepare-experiment", "evaluate-experiment"):
        experiment_main()
        return
    parser = argparse.ArgumentParser(
        description="Prepare the visual baseline dataset",
        epilog="Commands: prepare-reassemble, prepare-experiment, evaluate-experiment")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    inputs.add_argument("--dataset-plan", type=Path, help="Read only the selected REASSEMBLE recordings")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.dataset_plan is None:
        raw_dir = args.raw_dir.resolve()
        h5_files = sorted(raw_dir.rglob("*.h5"))
    else:
        plan = load_reassemble_plan(args.dataset_plan)
        if plan["camera"] != CAMERA:
            raise ValueError(f"Dataset plan must use camera {CAMERA}")
        raw_dir = Path(plan["raw_dir"])
        h5_files = recording_paths(raw_dir, plan["split"])
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
    if args.dataset_plan is not None:
        report["dataset_plan"] = str(args.dataset_plan.resolve())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = output_dir / f"dataset_report_{stamp}.json"
    write_json(report_path, report)
    print(f"Report: {report_path} ({report['recording_count']} recordings, {report['clip_count']} clips)")


def prepare_reassemble_main():
    '''Validate the official dataset selection and write its plan.'''
    parser = argparse.ArgumentParser(description="Prepare a plan from the official REASSEMBLE split")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Folder containing data and splits")
    parser.add_argument("--plan-file", type=Path, required=True, help="New dataset plan JSON path")
    parser.add_argument("--exclude-overlaps", action="store_true",
                        help="Record and exclude HDF5 recordings with overlapping annotations")
    args = parser.parse_args(sys.argv[2:])
    plan = prepare_reassemble(args.dataset_root, args.plan_file, args.exclude_overlaps, CAMERA)
    print(f"Saved: {args.plan_file} ({len(plan['split']['train'])} train, "
          f"{len(plan['split']['test'])} test, {len(plan['excluded'])} excluded)")


def experiment_main():
    '''
    Prepare reproducible few-shot experiments or evaluate one held-out partition.
    '''
    from src.experiment import prepare_experiment, evaluate_experiment
    from src.splitting import DEFAULT_CLASSES

    parser = argparse.ArgumentParser(description="Few-shot visual baseline experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-experiment", help="Freeze split and support selections")
    prepare.add_argument("--data-dir", type=Path, default=OUTPUT_DIR)
    prepare.add_argument("--experiment-dir", type=Path, required=True)
    prepare.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    prepare.add_argument("--shots", type=int, default=5)
    prepare.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    prepare.add_argument("--normalize-support", action="store_true",
                         help="Normalize each support embedding to unit length before averaging")
    prepare.add_argument("--split-seed", type=int, default=42)
    prepare.add_argument("--validation-fraction", type=float, default=0.2)
    prepare.add_argument("--test-fraction", type=float, default=0.2, help="Fraction for automatic splitting")
    split = prepare.add_mutually_exclusive_group()
    split.add_argument("--split-file", type=Path, help="Manual JSON with support/validation/test recording IDs")
    split.add_argument("--dataset-plan", type=Path, help="Use the official test IDs from a REASSEMBLE plan")
    evaluate = subparsers.add_parser("evaluate-experiment", help="Predict and save top-k and metrics")
    evaluate.add_argument("--experiment-dir", type=Path, required=True)
    evaluate.add_argument("--partition", choices=["validation", "test"], default="validation")
    evaluate.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.command == "prepare-experiment":
        plan = prepare_experiment(args.data_dir, args.experiment_dir, args.classes, args.shots,
                                  args.seeds, args.split_seed, args.validation_fraction,
                                  args.test_fraction, args.split_file, dataset_plan_file=args.dataset_plan,
                                  normalize_support=args.normalize_support)
        print("Split:", plan["split"])
        print("Segments per class:", plan["segment_counts"])
        print("Saved:", args.experiment_dir / "plan.json")
    else:
        summary = evaluate_experiment(args.experiment_dir, args.partition, args.top_k)
        print("Partition:", summary["partition"], "Queries:", summary["query_count_per_seed"])
        print("Metrics:", summary["metrics"])
        print("F1 by class:", summary["f1_by_class"])
        print("Top-k predictions:", args.experiment_dir / args.partition / "seed_<seed>.json")


if __name__ == "__main__":
    main()
