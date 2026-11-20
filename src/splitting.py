import itertools
import math

import numpy as np

PARTITIONS = ("support", "validation", "test")
DEFAULT_CLASSES = ("PICK", "INSERT", "REMOVE", "PLACE")


def segment_candidates(clips, classes):
    '''
    Group labeled windows by recording and segment, excluding IDLE and unknowns.
    '''
    groups = {}
    for clip in clips:
        if clip["ground_truth"] not in classes:
            continue
        if clip.get("segment_id") is None:
            raise ValueError(f"Labeled clip lacks segment ID: {clip['clip_id']}")
        key = (clip["recording_id"], clip["segment_id"])
        groups.setdefault(key, []).append(clip)
    for windows in groups.values():
        if len({clip["ground_truth"] for clip in windows}) != 1:
            raise ValueError("One segment contains conflicting activity labels")
    return groups


def validate_split(split, clips, classes, shots):
    '''
    Require disjoint recording partitions and sufficient class coverage.
    '''
    if set(split) != set(PARTITIONS):
        raise ValueError("Split must contain support, validation and test")
    assigned = []
    for partition in PARTITIONS:
        ids = split[partition]
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids):
            raise ValueError(f"{partition} must contain recording IDs")
        assigned.extend(ids)
    if len(assigned) != len(set(assigned)):
        raise ValueError("Recordings overlap across partitions or appear twice")
    available = {clip["recording_id"] for clip in clips}
    if set(assigned) != available:
        raise ValueError("Split must assign every available recording exactly once")
    groups = segment_candidates(clips, classes)
    counts = {}
    for partition in PARTITIONS:
        counts[partition] = {label: sum(
            windows[0]["ground_truth"] == label and record in split[partition]
            for (record, _), windows in groups.items()) for label in classes}
        minimum = shots if partition == "support" else 1
        missing = {label: count for label, count in counts[partition].items() if count < minimum}
        if missing:
            raise ValueError(f"Insufficient distinct segments in {partition}: {missing}; need {minimum} per class")
    return counts


def create_split(clips, classes, shots, seed=42, validation_fraction=0.2, test_fraction=0.2):
    '''
    Find a seeded recording split using label coverage, never embedding scores.
    Exhaust small search spaces and bound larger searches to 5000 candidates.
    '''
    if (not 0 < validation_fraction < 1 or not 0 < test_fraction < 1
            or validation_fraction + test_fraction >= 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    ids = sorted({clip["recording_id"] for clip in clips})
    if len(ids) < 3:
        raise ValueError("At least three distinct recordings are required")
    groups = segment_candidates(clips, classes)
    coverage = {label: len({record for (record, _), windows in groups.items()
                            if windows[0]["ground_truth"] == label}) for label in classes}
    if any(count < 3 for count in coverage.values()):
        raise ValueError(f"Each class needs at least three recordings for disjoint partitions: {coverage}")
    n_val = max(1, round(len(ids) * validation_fraction))
    n_test = max(1, round(len(ids) * test_fraction))
    if n_val + n_test >= len(ids):
        raise ValueError("Fractions leave no support recordings")
    rng = np.random.default_rng(seed)
    ids = list(rng.permutation(ids))
    space = math.comb(len(ids), n_test) * math.comb(len(ids) - n_test, n_val)
    if space <= 5000:
        candidates = ((list(test), list(val))
                      for test in itertools.combinations(ids, n_test)
                      for val in itertools.combinations([i for i in ids if i not in test], n_val))
    else:
        candidates = ((order[:n_test].tolist(), order[n_test:n_test+n_val].tolist())
                      for order in (rng.permutation(ids) for _ in range(5000)))
    for test, validation in candidates:
        split = {"support": [i for i in ids if i not in test and i not in validation],
                 "validation": validation, "test": test}
        try:
            validate_split(split, clips, classes, shots)
            return split
        except ValueError:
            continue
    raise ValueError("No valid split found at these fractions. Add recordings or provide a manual --split-file")


def split_with_fixed_test(train_ids, test_ids, seed, validation_fraction):
    '''Reserve validation recordings from train and keep the supplied test IDs.'''
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    validation_count = max(1, round(len(train_ids) * validation_fraction))
    if validation_count >= len(train_ids):
        raise ValueError("Official train split needs distinct support and validation recordings")

    # Sorting before sampling makes the split independent of the text file order.
    train = np.random.default_rng(seed).permutation(sorted(train_ids)).tolist()
    return {"support": sorted(train[validation_count:]),
            "validation": sorted(train[:validation_count]), "test": sorted(test_ids)}


def select_support(clips, recording_ids, classes, shots, seed):
    '''
    Sample distinct segments per class and take the central eligible window.
    '''
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer")
    groups = segment_candidates([c for c in clips if c["recording_id"] in recording_ids], classes)
    rng = np.random.default_rng(seed)
    selected = []
    for label in classes:
        keys = sorted(key for key, windows in groups.items() if windows[0]["ground_truth"] == label)
        if len(keys) < shots:
            raise ValueError(f"{label}: only {len(keys)} distinct support segments; need {shots}")
        for index in rng.choice(len(keys), size=shots, replace=False):
            windows = sorted(groups[keys[index]], key=lambda c: (c["start_frame"], c["clip_id"]))
            selected.append(windows[len(windows) // 2])
    return selected
