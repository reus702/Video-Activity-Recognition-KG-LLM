import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from src.batch import write_json
from src.evaluation import aggregate_metrics, classification_metrics
from src.inspection import load_embedding, load_recordings
from src.prototype_classifier import PrototypeClassifier
from src.splitting import create_split, select_support, validate_split


def file_digest(path):
    '''
    Hash file contents to detect changes after experiment preparation.
    '''
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_catalog(data_dir, classes, recording_names=None):
    '''
    Load eligible saved embeddings and preserve a fingerprint of their provenance.
    Reject duplicate IDs, multiple cameras per recording and mixed configurations.
    '''
    recordings = load_recordings(data_dir)
    if recording_names is not None:
        if not set(recording_names).issubset(recordings):
            raise ValueError("A prepared recording is missing")
        recordings = {name: recordings[name] for name in recording_names}
    catalog, snapshots, ids, recording_ids = {}, {}, set(), set()
    reference_config = None
    for name, recording in recordings.items():
        directory, manifest = recording["directory"], recording["manifest"]
        record_id = manifest["recording_id"]
        if record_id in recording_ids:
            raise ValueError("Use one camera/output per recording in this experiment")
        recording_ids.add(record_id)
        config = manifest["config"]
        if reference_config is None:
            reference_config = config
        elif config != reference_config:
            raise ValueError("Recording configurations differ; choose one consistent baseline run")
        snapshot = {"manifest_sha256": file_digest(directory / "manifest.json"),
                    "clips_sha256": file_digest(directory / "clips.json"), "embeddings": {}}
        for clip in recording["clips"]:
            if clip["recording_id"] != record_id:
                raise ValueError("Clip recording ID differs from its manifest")
            if clip["clip_id"] in ids:
                raise ValueError("Duplicate clip ID")
            ids.add(clip["clip_id"])
            if clip["ground_truth"] not in classes:
                continue
            vector = load_embedding(recording, clip)
            snapshot["embeddings"][clip["clip_id"]] = file_digest(directory / clip["embedding_path"])
            catalog[clip["clip_id"]] = {
                "clip": clip, "embedding": vector, "output_directory": name
            }
        snapshots[name] = snapshot
    if not catalog:
        raise ValueError("No clips from the requested activity classes")
    return catalog, snapshots, reference_config, recording_ids


def prepare_experiment(data_dir, experiment_dir, classes, shots, seeds, split_seed=42,
                       validation_fraction=0.2, test_fraction=0.2, split_file=None):
    '''
    Freeze recording splits and support selections in a new experiment folder.
    Do not classify queries or process raw HDF5 files.
    '''
    classes, seeds = list(classes), list(seeds)
    if (not classes or len(classes) != len(set(classes))
            or any(not isinstance(label, str) or not label.strip() for label in classes)
            or "IDLE" in classes or "UNLABELED" in classes):
        raise ValueError("Use unique nonempty activity classes, excluding IDLE and UNLABELED")
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer")
    if (not seeds or len(seeds) != len(set(seeds))
            or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)
            or split_seed < 0):
        raise ValueError("Use distinct nonnegative integer seeds")
    experiment_dir = Path(experiment_dir)
    if experiment_dir.exists():
        raise FileExistsError(f"Experiment already exists: {experiment_dir}")
    catalog, snapshots, config, recording_ids = build_catalog(data_dir, classes)
    clips = [item["clip"] for item in catalog.values()]
    # Recordings without eligible clips cannot contribute to these partitions.
    eligible_ids = {clip["recording_id"] for clip in clips}
    if eligible_ids != recording_ids:
        raise ValueError(f"Recordings without requested classes: {sorted(recording_ids - eligible_ids)}")
    if split_file is not None:
        split = json.loads(Path(split_file).read_text())
    else:
        split = create_split(clips, classes, shots, split_seed, validation_fraction, test_fraction)
    counts = validate_split(split, clips, classes, shots)
    supports = {str(seed): [c["clip_id"] for c in select_support(
        clips, split["support"], classes, shots, seed)] for seed in seeds}
    plan = {
        "schema_version": 1, "data_dir": str(Path(data_dir).resolve()),
        "classes": classes, "shots": shots, "seeds": seeds, "split_seed": split_seed,
        "split_method": "manual" if split_file else "seeded_class_coverage_search",
        "requested_fractions": {"validation": validation_fraction, "test": test_fraction},
        "split": split, "segment_counts": counts, "support_clip_ids": supports,
        "snapshot": snapshots, "embedding_config": config,
        "classifier": "mean_raw_embeddings_then_cosine",
        "support_policy": "distinct segments; central eligible window per chosen segment",
        "query_policy": "all labeled windows in held-out recordings",
        "excluded": ["IDLE", "unlabeled", "classes outside the requested list"],
        "caveat": "Query windows can overlap; window-level metrics do not treat them as independent trials.",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__}
    }
    experiment_dir.mkdir(parents=True)
    write_json(experiment_dir / "plan.json", plan)
    return plan


def evaluate_experiment(experiment_dir, partition="validation", top_k=3):
    '''
    Fit saved support selections and evaluate one explicitly chosen partition.
    Refuse changed inputs and never overwrite existing evaluation results.
    '''
    if partition not in ("validation", "test"):
        raise ValueError("Choose validation or test")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    experiment_dir = Path(experiment_dir)
    result_dir = experiment_dir / partition
    if result_dir.exists():
        raise FileExistsError(f"Results already exist: {result_dir}")
    plan = json.loads((experiment_dir / "plan.json").read_text())
    catalog, snapshots, _, _ = build_catalog(plan["data_dir"], plan["classes"], plan["snapshot"])
    if snapshots != plan["snapshot"]:
        raise ValueError("Prepared metadata or embeddings changed; create a new experiment")
    clips = [item["clip"] for item in catalog.values()]
    validate_split(plan["split"], clips, plan["classes"], plan["shots"])
    queries = sorted((item for item in catalog.values()
                      if item["clip"]["recording_id"] in plan["split"][partition]),
                     key=lambda item: item["clip"]["clip_id"])
    results = []
    for seed in plan["seeds"]:
        # Recompute the selection to catch accidental plan edits and leakage.
        expected = [c["clip_id"] for c in select_support(
            clips, plan["split"]["support"], plan["classes"], plan["shots"], seed)]
        if expected != plan["support_clip_ids"][str(seed)]:
            raise ValueError("Saved support selection does not match the plan")
        support = [catalog[clip_id] for clip_id in expected]
        classifier = PrototypeClassifier()
        classifier.fit(np.stack([item["embedding"] for item in support]),
                       [item["clip"]["ground_truth"] for item in support])
        predictions = []
        for item in queries:
            clip = item["clip"]
            # Keep a full ranking for metrics even when fewer candidates are displayed.
            ranked = classifier.predict(item["embedding"], top_k=len(plan["classes"]))
            predictions.append({**clip, "output_directory": item["output_directory"],
                                "predicted_activity": ranked[0]["action"],
                                "top_k": ranked[:top_k], "candidates": ranked})
        results.append({"seed": seed, "predictions": predictions,
                        "metrics": classification_metrics(predictions, plan["classes"]),
                        "prototypes": {label: vector.tolist() for label, vector in classifier.prototypes.items()}})
    # Save only after all computations succeed. A partial folder is never reused.
    result_dir.mkdir()
    for result in results:
        write_json(result_dir / f"seed_{result['seed']}.json", result)
    summary = {"partition": partition, "seeds": plan["seeds"], "display_top_k": top_k,
               "plan_sha256": file_digest(experiment_dir / "plan.json"),
               "metrics": aggregate_metrics([r["metrics"] for r in results]),
               "query_count_per_seed": len(queries), "class_order": plan["classes"],
               "caveat": plan["caveat"], "complete": True}
    write_json(result_dir / "summary.json", summary)
    return summary
