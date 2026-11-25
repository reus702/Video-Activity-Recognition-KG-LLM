import json
from collections import defaultdict
from pathlib import Path
from time import monotonic

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18

from src.batch import write_json
from src.experiment import file_digest
from src.feature_extraction import VIDEO_WEIGHTS
from src.inspection import previous_clips
from src.video_training import SavedVideoClips, load_training_clips, print_progress, rank_predictions


@torch.inference_mode()
def predict_validation_context(run_dir, output_path):
    '''Score the missing predecessors of saved validation predictions.'''
    run_dir, output_path = Path(run_dir), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Context predictions already exist: {output_path}")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    plan_path = Path(run["source_plan"])
    if not summary["complete"] or file_digest(plan_path) != run["source_plan_sha256"]:
        raise ValueError("Incomplete training run or modified source plan")
    if run["weights"] != str(VIDEO_WEIGHTS):
        raise ValueError("Training run uses different video preprocessing")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    predictions_path = run_dir / "validation_predictions.json"
    saved = json.loads(predictions_path.read_text(encoding="utf-8"))
    saved_ids = {row["clip_id"] for row in saved}
    if len(saved_ids) != len(saved) or saved_ids != {row["clip_id"] for row in run["validation_clips"]}:
        raise ValueError("Saved predictions do not match the training run's validation clips")
    _, clips = load_training_clips(plan, partitions=("validation",))
    clips_by_recording = defaultdict(list)
    for clip in clips:
        clips_by_recording[clip["recording_id"]].append(clip)
    if not saved_ids.issubset(clip["clip_id"] for clip in clips):
        raise ValueError("Saved predictions include clips outside the validation timeline")

    predecessors = {}
    for timeline in clips_by_recording.values():
        predecessors.update(previous_clips(timeline))
    missing = {}
    already_saved = no_previous_window = 0
    for row in saved:
        previous = predecessors.get(row["clip_id"])
        if previous is None:
            no_previous_window += 1
        elif previous["clip_id"] in saved_ids:
            already_saved += 1
        else:
            missing[previous["clip_id"]] = previous

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classes = run["classes"]
    if checkpoint["epoch"] != summary["best_epoch"] or checkpoint["classes"] != classes:
        raise ValueError("Checkpoint epoch or classes do not match the saved training run")
    model = r2plus1d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)

    selected = list(missing.values())
    videos = {}
    for recording_id in sorted({clip["recording_id"] for clip in selected}):
        source = run["videos"][recording_id]
        path = Path(source["path"])
        if file_digest(path) != source["sha256"]:
            raise ValueError(f"Source video changed: {path}")
        videos[recording_id] = path

    loader = DataLoader(SavedVideoClips(selected, videos), batch_size=run["batch_size"])
    print(f"Checkpoint epoch {checkpoint['epoch']} | inference on {device} | "
          f"{len(selected)} missing context clips", flush=True)
    started = monotonic()
    scores = []
    for step, frames in enumerate(loader, start=1):
        scores.extend(model(frames.to(device)).softmax(dim=1).cpu().numpy())
        print_progress("Context inference batch", step, len(loader), started)
    predictions = rank_predictions(selected, scores, classes)
    report = {"schema_version": 1, "partition": "validation", "checkpoint_epoch": checkpoint["epoch"],
              "checkpoint_sha256": file_digest(checkpoint_path), "source_plan_sha256": run["source_plan_sha256"],
              "validation_predictions_sha256": file_digest(predictions_path),
              "classes": classes, "device": str(device), "torch_version": str(torch.__version__),
              "validation_clip_count": len(saved), "previous_prediction_already_saved": already_saved,
              "no_previous_window": no_previous_window, "prediction_count": len(predictions),
              "predictions": predictions}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    return report
