"""Estimate gripper opening from the saved wrist-camera RGB frames."""

import json
from math import isfinite
from pathlib import Path

import cv2

from src.experiment import file_digest
from src.video_processing import read_sampled_frames


# Template and thresholds chosen on the support recordings, in resized pixels.
TEMPLATE_RECORDING = "2025-01-09-14-28-47"
TEMPLATE_FRAME = 6225
FRAME_SIZE = (320, 180)
JAW_CROPS = ((108, 128), (182, 202))
SEARCH_STRIPS = ((14, 149), (156, 306))
MIN_CORRELATION = 0.78
MAX_VERTICAL_OFFSET = 6
EXPECTED_MIDPOINT = 145
MAX_MIDPOINT_OFFSET = 8
CLOSED_MAX_DISTANCE = 90
OPEN_MIN_DISTANCE = 220


def wrist_grayscale(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), FRAME_SIZE,
                      interpolation=cv2.INTER_AREA)


class GripperOpeningEstimator:
    """Locate each jaw in its search strip and abstain on intermediate gaps."""

    def __init__(self, template_frame):
        self.frame_shape = template_frame.shape
        image = wrist_grayscale(template_frame)
        self.templates = [image[101:128, left:right].copy() for left, right in JAW_CROPS]

    def observe(self, frame):
        if frame.shape != self.frame_shape:
            raise ValueError("Wrist frame geometry differs from the gripper template")
        image = wrist_grayscale(frame)
        matches = []
        for (left, right), template in zip(SEARCH_STRIPS, self.templates, strict=True):
            scores = cv2.matchTemplate(image[96:137, left:right], template, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            matches.append((left + x, y, score))

        (left_x, left_y, left_score), (right_x, right_y, right_score) = matches
        distance = right_x - left_x
        correlation = min(left_score, right_score)
        midpoint = (left_x + right_x) / 2
        opening = None
        if (correlation >= MIN_CORRELATION
                and abs(left_y - right_y) <= MAX_VERTICAL_OFFSET
                and abs(midpoint - EXPECTED_MIDPOINT) <= MAX_MIDPOINT_OFFSET):
            if distance <= CLOSED_MAX_DISTANCE:
                opening = "closed"
            elif distance >= OPEN_MIN_DISTANCE:
                opening = "open"
        return {
            "gripper_opening": opening,
            "jaw_match_distance_px": distance,
            "min_template_correlation": correlation,
        }


def load_validation_wrist_clips(plan_path, clip_ids):
    """Resolve the saved RGB inputs and verify their validation membership and samples."""
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("input_type") != "rgb" or plan.get("camera") != "hand":
        raise ValueError("Expected a wrist-camera RGB plan")
    split = plan["split"]
    assigned = [recording for recordings in split.values() for recording in recordings]
    if len(assigned) != len(set(assigned)) or TEMPLATE_RECORDING not in split["support"]:
        raise ValueError("Expected disjoint partitions with the template recording in support")
    if not clip_ids or len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Provide distinct validation clip IDs")

    # The exported videos sit beside their plan even after the experiment is moved.
    root = plan_path.parent
    validation_ids = {f"{recording}_hand": recording for recording in split["validation"]}
    rows = []
    for clip_id in clip_ids:
        video_id, _, _ = clip_id.rpartition("_clip_")
        if video_id not in validation_ids:
            raise ValueError(f"Expected a validation clip: {clip_id}")
        directory = root / video_id
        clips_path = directory / "clips.json"
        if file_digest(clips_path) != plan["snapshot"][video_id]["clips_sha256"]:
            raise ValueError(f"Clip metadata differs from the RGB plan: {video_id}")
        clips = json.loads(clips_path.read_text(encoding="utf-8"))
        matching = [clip for clip in clips if clip["clip_id"] == clip_id]
        if len(matching) != 1:
            raise ValueError(f"Expected one saved clip for {clip_id}")
        clip = matching[0]
        if clip["recording_id"] != validation_ids[video_id] or clip["camera"] != "hand":
            raise ValueError(f"Clip metadata does not match the requested wrist recording: {clip_id}")
        indices = clip["sampled_frame_indices"]
        timestamps = clip["original_frame_times"]
        if (len(indices) != len(timestamps) or len(indices) < 2
                or not all(isfinite(timestamp) for timestamp in timestamps)
                or any(first >= second for first, second in zip(indices, indices[1:]))
                or any(first >= second for first, second in zip(timestamps, timestamps[1:]))
                or not clip["start_frame"] <= indices[0] < indices[-1] < clip["end_frame_exclusive"]):
            raise ValueError(f"Invalid saved samples for {clip_id}")
        rows.append((directory / f"{video_id}.mp4", clip))

    template_id = f"{TEMPLATE_RECORDING}_hand"
    template_video = root / template_id / f"{template_id}.mp4"
    return template_video, rows


def observe_validation_gripper(plan_path, clip_ids):
    """Use RGB and frozen frame metadata only. Return two observations per clip."""
    template_video, rows = load_validation_wrist_clips(plan_path, clip_ids)
    template_frame, = read_sampled_frames(template_video, [TEMPLATE_FRAME])
    estimator = GripperOpeningEstimator(template_frame)
    observations = []
    for video, clip in rows:
        indices = clip["sampled_frame_indices"]
        timestamps = clip["original_frame_times"]
        frames = read_sampled_frames(video, [indices[0], indices[-1]])
        for endpoint, frame, slot in zip(("first", "last"), frames, (0, -1), strict=True):
            observations.append({
                "clip_id": clip["clip_id"], "recording_id": clip["recording_id"],
                "segment_id": clip["segment_id"], "camera": "hand",
                "endpoint": endpoint, "frame_index": indices[slot], "observed_at": timestamps[slot],
                **estimator.observe(frame),
            })
    return {
        "method": "wrist_rgb_jaw_template_matching_v1",
        "source_plan_sha256": file_digest(plan_path),
        "template": {
            "recording_id": TEMPLATE_RECORDING, "frame_index": TEMPLATE_FRAME,
            "video_sha256": file_digest(template_video),
        },
        "observations": observations,
    }
