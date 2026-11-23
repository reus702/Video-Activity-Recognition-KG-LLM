import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r2plus1d_18

from src.batch import write_json
from src.evaluation import classification_metrics
from src.experiment import file_digest
from src.feature_extraction import VIDEO_WEIGHTS, preprocess_frames
from src.inspection import load_embedding
from src.prototype_classifier import PrototypeClassifier
from src.splitting import select_support
from src.video_processing import read_sampled_frames


def load_training_clips(plan):
    '''Load metadata from the support and validation recordings in a frozen plan.'''
    split = plan["split"]
    assigned = [recording for ids in split.values() for recording in ids]
    if len(assigned) != len(set(assigned)):
        raise ValueError("Recordings overlap across experiment partitions")
    requested = set(split["support"]) | set(split["validation"])
    root = Path(plan["data_dir"])
    recordings, clips = {}, []
    for name, snapshot in plan["snapshot"].items():
        directory = root / name
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recording_id = manifest["recording_id"]
        if recording_id not in requested:
            continue
        clips_path = directory / "clips.json"
        if (recording_id in recordings or not manifest["complete"]
                or file_digest(manifest_path) != snapshot["manifest_sha256"]
                or file_digest(clips_path) != snapshot["clips_sha256"]):
            raise ValueError(f"Visual metadata changed or is ambiguous for {name}")
        saved_clips = json.loads(clips_path.read_text(encoding="utf-8"))
        if any(clip["recording_id"] != recording_id for clip in saved_clips):
            raise ValueError(f"Clip recording ID does not match {name}")
        recordings[recording_id] = {"directory": directory, "manifest": manifest}
        clips.extend(saved_clips)
    if set(recordings) != requested:
        raise ValueError("Saved recordings do not match the support and validation split")
    return recordings, clips


def select_training_clips(clips, recording_ids, classes, shots, seed):
    '''Use all eligible windows or one central window per sampled segment.'''
    if shots != "all":
        return select_support(clips, recording_ids, classes, shots, seed)

    recording_ids = set(recording_ids)
    selected = [clip for clip in clips
                if clip["recording_id"] in recording_ids and clip["ground_truth"] in classes]
    missing = set(classes) - {clip["ground_truth"] for clip in selected}
    if missing:
        raise ValueError(f"No labeled windows for classes: {sorted(missing)}")
    return selected


class SavedVideoClips(Dataset):
    '''Decode the exact RGB samples used to create each clip's original embedding.'''

    def __init__(self, selected, videos, classes):
        self.clips = selected
        self.videos = videos
        self.class_index = {label: index for index, label in enumerate(classes)}
        self.transform = VIDEO_WEIGHTS.transforms()

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip = self.clips[index]
        frames = read_sampled_frames(self.videos[clip["recording_id"]], clip["sampled_frame_indices"])
        return preprocess_frames(frames, self.transform), self.class_index[clip["ground_truth"]]


def frozen_comparison(plan, recordings, training, validation):
    '''Fit the frozen prototype control on the same examples as the trained model.'''
    vectors = {}
    for clip in training + validation:
        recording = recordings[clip["recording_id"]]
        directory = recording["directory"]
        expected = plan["snapshot"][directory.name]["embeddings"][clip["clip_id"]]
        if file_digest(directory / clip["embedding_path"]) != expected:
            raise ValueError(f"Frozen embedding changed: {clip['clip_id']}")
        vectors[clip["clip_id"]] = load_embedding(recording, clip)
    classifier = PrototypeClassifier()
    classifier.fit(np.stack([vectors[clip["clip_id"]] for clip in training]),
                   [clip["ground_truth"] for clip in training])
    predictions = [{"ground_truth": clip["ground_truth"],
                    "candidates": classifier.predict(vectors[clip["clip_id"]], len(plan["classes"]))}
                   for clip in validation]
    return classification_metrics(predictions, plan["classes"])


@torch.inference_mode()
def validate_model(model, loader, criterion, classes, device):
    '''Score the fixed validation windows in their saved order.'''
    model.eval()
    loss = 0.0
    scores = []
    for frames, labels in loader:
        logits = model(frames.to(device))
        loss += criterion(logits, labels.to(device)).item() * len(labels)
        scores.extend(logits.softmax(dim=1).cpu().numpy())
    predictions = []
    for clip, probabilities in zip(loader.dataset.clips, scores, strict=True):
        ranking = np.argsort(-probabilities, kind="stable")
        predictions.append({**clip, "predicted_activity": classes[ranking[0]],
                            "candidates": [{"action": classes[index], "score": float(probabilities[index])}
                                           for index in ranking]})
    return loss / len(predictions), classification_metrics(predictions, classes), predictions


def train_visual(plan_path, output_dir, train_shots=32, validation_shots=16, epochs=3,
                 batch_size=4, seed=0, learning_rate=1e-4):
    '''Fine-tune the last R(2+1)D block and the action head on labeled support videos.'''
    if min(epochs, batch_size) < 1 or seed < 0:
        raise ValueError("Use positive epoch and batch counts and a nonnegative seed")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    plan_path, output_dir = Path(plan_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Training output already exists: {output_dir}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    classes = plan["classes"]
    recordings, clips = load_training_clips(plan)
    training = select_training_clips(clips, plan["split"]["support"], classes, train_shots, seed)
    # Keep validation clip selection fixed across training seeds.
    validation = select_training_clips(clips, plan["split"]["validation"], classes,
                                       validation_shots, plan["split_seed"])
    videos = {}
    for recording_id in sorted({clip["recording_id"] for clip in training + validation}):
        directory = recordings[recording_id]["directory"]
        videos[recording_id] = directory / f"{directory.name}.mp4"
    video_hashes = {recording_id: file_digest(path) for recording_id, path in videos.items()}
    baseline = frozen_comparison(plan, recordings, training, validation)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(SavedVideoClips(training, videos, classes), batch_size=batch_size,
                              shuffle=True, generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(SavedVideoClips(validation, videos, classes), batch_size=batch_size)
    model = r2plus1d_18(weights=VIDEO_WEIGHTS)
    model.requires_grad_(False)
    model.layer4.requires_grad_(True)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([{"params": model.layer4.parameters(), "lr": learning_rate},
                                   {"params": model.fc.parameters(), "lr": learning_rate * 10}])
    output_dir.mkdir(parents=True)
    write_json(output_dir / "run.json", {
        "source_plan": str(plan_path.resolve()), "source_plan_sha256": file_digest(plan_path),
        "weights": str(VIDEO_WEIGHTS), "torch_version": str(torch.__version__), "device": str(device),
        "trainable_layers": ["layer4", "fc"], "optimizer": "AdamW",
        "batch_size": batch_size, "epochs": epochs, "seed": seed, "validation_seed": plan["split_seed"],
        "train_shots": train_shots, "validation_shots": validation_shots,
        "learning_rate": learning_rate, "head_learning_rate": learning_rate * 10,
        "classes": classes, "train_clips": training, "validation_clips": validation,
        "videos": {recording_id: {"path": str(path), "sha256": video_hashes[recording_id]}
                   for recording_id, path in videos.items()},
    })
    print(f"Training on {device}: {len(training)} train clips, {len(validation)} validation clips",
          flush=True)

    best_epoch, best_metrics, best_predictions, best_weights = None, None, None, None
    for epoch in range(1, epochs + 1):
        model.eval()  # Keep frozen backbone batch-normalization statistics unchanged.
        model.layer4.train()
        model.fc.train()
        loss = 0.0
        for step, (frames, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            step_loss = criterion(model(frames.to(device)), labels.to(device))
            if not torch.isfinite(step_loss):
                raise ValueError("Training loss is nonfinite")
            step_loss.backward()
            optimizer.step()
            loss += step_loss.item() * len(labels)
            if step % 250 == 0:
                print(f"Epoch {epoch}/{epochs} | training batch {step}/{len(train_loader)}", flush=True)
        training_loss = loss / len(training)
        validation_loss, metrics, predictions = validate_model(
            model, validation_loader, criterion, classes, device)
        write_json(output_dir / f"epoch_{epoch:02d}.json", {
            "epoch": epoch, "training_loss": training_loss, "validation_loss": validation_loss,
            "metrics": metrics,
        })
        print(f"Epoch {epoch}/{epochs} | train loss {training_loss:.4f} | "
              f"val loss {validation_loss:.4f} | val Macro-F1 {metrics['macro_f1']:.4f}", flush=True)
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_epoch, best_metrics, best_predictions = epoch, metrics, predictions
            best_weights = {name: weight.detach().cpu().clone() for name, weight in model.state_dict().items()}

    with (output_dir / "checkpoint.pt").open("xb") as stream:
        torch.save({"model_state_dict": best_weights, "classes": classes, "epoch": best_epoch}, stream)
    write_json(output_dir / "validation_predictions.json", best_predictions)
    summary = {"best_epoch": best_epoch, "metrics": best_metrics, "frozen_prototype_metrics": baseline,
               "train_clip_count": len(training), "validation_clip_count": len(validation), "complete": True}
    write_json(output_dir / "summary.json", summary)
    return summary
