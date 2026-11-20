import json
from pathlib import Path

from src.batch import write_json


OFFICIAL_PARTITIONS = ("train", "test")


def validate_official_split(split):
    '''Require nonempty, disjoint lists of recording IDs from the official split.'''
    if not isinstance(split, dict) or set(split) != set(OFFICIAL_PARTITIONS):
        raise ValueError("REASSEMBLE split must contain train and test")
    for partition in OFFICIAL_PARTITIONS:
        ids = split[partition]
        if not isinstance(ids, list) or not ids or any(
                not isinstance(recording_id, str) or not recording_id
                or recording_id in (".", "..")
                or Path(recording_id).name != recording_id for recording_id in ids):
            raise ValueError(f"Invalid recording IDs in REASSEMBLE {partition}")
    ids = split["train"] + split["test"]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate recording IDs within or across official splits")


def recording_paths(raw_dir, split):
    '''Resolve only the HDF5 files assigned by the plan.'''
    paths = [raw_dir / f"{recording_id}.h5" for recording_id in sorted(split["train"] + split["test"])]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing REASSEMBLE recording: {path}")
    return paths


def prepare_reassemble(dataset_root, plan_path, exclude_overlaps=False, camera="hama1"):
    '''Check official recordings and save the selection before running the encoder.'''
    from src.labeling import OverlappingSegmentsError, load_annotations

    dataset_root = Path(dataset_root).resolve()
    plan_path = Path(plan_path)
    if plan_path.exists():
        raise FileExistsError(f"Dataset plan already exists: {plan_path}")
    raw_dir = dataset_root / "data"
    official = {}
    for partition in OFFICIAL_PARTITIONS:
        split_path = dataset_root / "splits" / f"{partition}_split1.txt"
        official[partition] = split_path.read_text(encoding="utf-8-sig").split()
    validate_official_split(official)
    paths = {path.stem: path for path in recording_paths(raw_dir, official)}

    selected = {partition: [] for partition in OFFICIAL_PARTITIONS}
    excluded = {}
    for partition in OFFICIAL_PARTITIONS:
        for recording_id in sorted(official[partition]):
            path = paths[recording_id]
            try:
                load_annotations(path, camera)
            except OverlappingSegmentsError as exc:
                if not exclude_overlaps:
                    raise OverlappingSegmentsError(f"{path}: {exc}") from exc
                excluded[recording_id] = {"partition": partition, "reason": str(exc)}
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot validate {path}: {exc}") from exc
            else:
                selected[partition].append(recording_id)

    validate_official_split(selected)
    plan = {"schema_version": 1, "dataset": "REASSEMBLE", "raw_dir": str(raw_dir),
            "camera": camera, "split": selected, "excluded": excluded}
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, plan)
    return plan


def load_reassemble_plan(path):
    '''Read a saved plan without accessing the raw recordings.'''
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(plan, dict) or plan.get("dataset") != "REASSEMBLE"
            or plan.get("schema_version") != 1 or not isinstance(plan.get("raw_dir"), str)
            or not Path(plan["raw_dir"]).is_absolute() or not isinstance(plan.get("camera"), str)
            or not isinstance(plan.get("excluded"), dict)):
        raise ValueError(f"Invalid REASSEMBLE dataset plan: {path}")
    validate_official_split(plan.get("split"))
    return plan
