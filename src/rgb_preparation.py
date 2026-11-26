import json
from collections import Counter
from pathlib import Path

from src.batch import source_identity, write_json
from src.dataset import extract_video
from src.experiment import file_digest
from src.labeling import label_clips, load_annotations, validate_video
from src.reassemble import recording_paths, validate_official_split
from src.splitting import validate_split
from src.video_processing import iter_clip_metadata


RGB_CAMERAS = ("hama1", "hama2", "hand")


def prepare_rgb(source_plan_path, output_dir, camera):
    """Extract one camera and write a training plan that needs only RGB and labels."""
    if camera not in RGB_CAMERAS:
        raise ValueError(f"Expected an RGB camera from {RGB_CAMERAS}")
    source_plan_path, output_dir = Path(source_plan_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"RGB output already exists: {output_dir}")
    source = json.loads(source_plan_path.read_text(encoding="utf-8"))
    dataset = source.get("dataset_plan", {})
    if dataset.get("dataset") != "REASSEMBLE":
        raise ValueError("Use a visual experiment plan prepared from the official REASSEMBLE split")
    official = dataset["split"]
    validate_official_split(official)
    split = source["split"]
    if (set(split) != {"support", "validation", "test"}
            or sorted(split["support"] + split["validation"]) != sorted(official["train"])
            or sorted(split["test"]) != sorted(official["test"])):
        raise ValueError("Source experiment split differs from the selected official recordings")

    settings = {key: source["embedding_config"][key] for key in ("window_sec", "stride_sec", "num_frames")}
    paths = recording_paths(Path(dataset["raw_dir"]), official)
    output_dir.mkdir(parents=True)
    all_clips, snapshots = [], {}
    for index, path in enumerate(paths, start=1):
        print(f"Preparing {camera} | recording {index}/{len(paths)} | {path.stem}", flush=True)
        identity = source_identity(path)
        _, timestamps = load_annotations(path, camera)
        video_id = f"{path.stem}_{camera}"
        directory = output_dir / video_id
        video = extract_video(path, directory, camera)
        fps, total = validate_video(video, timestamps)
        windows = [
            {**clip, "clip_id": f"{video_id}_clip_{i:04d}", "video_id": video_id,
             "recording_id": path.stem, "camera": camera}
            for i, clip in enumerate(iter_clip_metadata(fps, total, **settings))
        ]
        if not windows:
            raise ValueError(f"No complete windows for {video_id}")
        clips = label_clips(path, video, windows, camera)
        if source_identity(path) != identity:
            raise ValueError(f"Source changed during extraction: {path}")

        clips_path, manifest_path = directory / "clips.json", directory / "manifest.json"
        write_json(clips_path, clips)
        write_json(manifest_path, {
            "complete": True, "input_type": "rgb", "recording_id": path.stem, "camera": camera,
            "source": {**identity, "relative_path": path.name}, "config": settings, "clip_count": len(clips),
            "video_fps": fps, "video_frame_count": total,
        })
        snapshots[video_id] = {"manifest_sha256": file_digest(manifest_path),
                               "clips_sha256": file_digest(clips_path)}
        all_clips.extend(clips)

    classes = source["classes"]
    segment_counts = validate_split(split, all_clips, classes, shots=1)
    partition_by_recording = {recording: partition for partition, ids in split.items() for recording in ids}
    counts = Counter((partition_by_recording[c["recording_id"]], c["ground_truth"]) for c in all_clips)
    plan = {
        "schema_version": 1, "input_type": "rgb", "camera": camera, "data_dir": str(output_dir),
        "source_plan": str(source_plan_path), "source_plan_sha256": file_digest(source_plan_path),
        "dataset_plan": {**dataset, "camera": camera}, "split": split, "split_seed": source["split_seed"],
        "classes": classes, "sampling": settings, "snapshot": snapshots, "segment_counts": segment_counts,
        "clip_counts": {partition: {label: counts[partition, label] for label in classes} for partition in split},
    }
    # Publish the plan last, after checking class coverage on the extracted camera.
    write_json(output_dir / "plan.json", plan)
    return plan
