import json
from bisect import bisect_right
from pathlib import Path

import numpy as np


def previous_clips(clips):
    '''Find the latest non-overlapping predecessor within one recording.'''
    ordered = sorted(clips, key=lambda clip: clip["end_frame_exclusive"])
    ends = [clip["end_frame_exclusive"] for clip in ordered]
    previous = {}
    for clip in clips:
        index = bisect_right(ends, clip["start_frame"]) - 1
        if index >= 0:
            previous[clip["clip_id"]] = ordered[index]
    return previous


def load_recordings(output_dir):
    '''
    Load completed recording metadata without modifying generated outputs.
    Embedding paths remain relative to each recording directory.
    '''
    recordings = {}
    for manifest_path in sorted(Path(output_dir).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete"):
            continue
        directory = manifest_path.parent
        clips = json.loads((directory / "clips.json").read_text())
        if len(clips) != manifest["clip_count"]:
            raise ValueError(f"Clip count differs from manifest: {directory}")
        recordings[directory.name] = {
            "directory": directory, "manifest": manifest, "clips": clips
        }
    if not recordings:
        raise FileNotFoundError(f"No completed recordings in {output_dir}; run main.py first")
    return recordings


def load_embedding(recording, clip):
    '''
    Load a clip embedding using its recording directory and check its values.
    '''
    embedding = np.load(recording["directory"] / clip["embedding_path"], allow_pickle=False)
    expected = recording["manifest"]["config"]["embedding_dim"]
    if (embedding.shape != (expected,) or not np.isfinite(embedding).all()
            or np.linalg.norm(embedding) == 0):
        raise ValueError(f"Invalid embedding: {clip['clip_id']}")
    return embedding


def detection_rows(clips):
    '''
    Flatten optional per-frame detections for display without inventing objects.
    Empty detections produce no rows; annotation objects remain separate.
    '''
    rows = []
    for clip in clips:
        for frame in clip.get("detections", []) or []:
            for obj in frame.get("objects", []) or []:
                rows.append({
                    "recording_id": clip["recording_id"], "clip_id": clip["clip_id"],
                    "timestamp": frame["timestamp"], "label": obj["label"],
                    "confidence": obj["confidence"], "bbox": obj["bbox"]
                })
    return rows


def table_html(rows, columns=None):
    '''
    Format inspection rows as an escaped HTML table without requiring pandas.
    '''
    from html import escape

    rows = list(rows)
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    header = "".join(f"<th>{escape(str(key))}</th>" for key in columns)
    body = "".join("<tr>" + "".join(
        f"<td>{escape(str(row.get(key, '')))}</td>" for key in columns
    ) + "</tr>" for row in rows)
    if not rows:
        body = f'<tr><td colspan="{max(1, len(columns))}">No rows available.</td></tr>'
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
