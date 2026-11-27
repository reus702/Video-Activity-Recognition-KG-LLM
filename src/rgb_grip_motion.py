"""Measure texture moving with the wrist relative to the surrounding scene."""

import cv2
import numpy as np

from src.experiment import file_digest
from src.rgb_gripper import (
    TEMPLATE_FRAME,
    TEMPLATE_RECORDING,
    GripperOpeningEstimator,
    load_validation_wrist_clips,
    wrist_grayscale,
)
from src.video_processing import read_sampled_frames


def _track_points(before, after, region):
    points = cv2.goodFeaturesToTrack(before, maxCorners=120, qualityLevel=.015,
                                    minDistance=3, mask=region)
    if points is None:
        return np.empty((0, 2)), np.empty((0, 2))
    options = {"winSize": (15, 15), "maxLevel": 2}
    tracked, forward, error = cv2.calcOpticalFlowPyrLK(before, after, points, None, **options)
    returned, backward, _ = cv2.calcOpticalFlowPyrLK(after, before, tracked, None, **options)
    round_trip = np.linalg.norm(returned[:, 0] - points[:, 0], axis=1)
    valid = (forward[:, 0] > 0) & (backward[:, 0] > 0) & (error[:, 0] < 18) & (round_trip < 1.5)
    return points[valid, 0], tracked[valid, 0]


def wrist_motion_cue(before, after):
    """Return diagnostics for a possible held piece in two normalized grayscale frames."""
    background = np.zeros_like(before)
    background[8:92, 20:117] = 255
    background[8:92, 192:300] = 255
    start, end = _track_points(before, after, background)
    if len(start) < 12:
        return {"cue": None, "reason": "too_few_scene_tracks"}
    transform, inliers = cv2.estimateAffinePartial2D(
        start, end, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if transform is None or int(inliers.sum()) < 10 or inliers.mean() < .55:
        return {"cue": None, "reason": "inconsistent_scene_motion"}

    center = np.array([156, 108])
    scene_motion = float(np.linalg.norm(transform @ [*center, 1] - center))
    if scene_motion < 3:
        return {"cue": None, "reason": "insufficient_scene_motion", "scene_motion_px": scene_motion}

    between_jaws = np.zeros_like(before)
    between_jaws[88:124, 146:166] = 255
    start, end = _track_points(before, after, between_jaws)
    track_count = len(start)
    stationary = int(np.count_nonzero(np.linalg.norm(end - start, axis=1) < 1))
    detected = stationary >= 4 and 3 * stationary >= 2 * track_count
    return {
        "cue": "possible_held_piece" if detected else None,
        "reason": "texture_stationary_in_wrist_view" if detected else "insufficient_stationary_texture",
        "scene_motion_px": scene_motion,
        "center_track_count": track_count,
        "stationary_center_tracks": stationary,
    }


def observe_validation_grip_motion(plan_path, clip_ids):
    """Evaluate adjacent saved samples without looking outside the clip."""
    template_video, rows = load_validation_wrist_clips(plan_path, clip_ids)
    template_frame, = read_sampled_frames(template_video, [TEMPLATE_FRAME])
    estimator = GripperOpeningEstimator(template_frame)
    observations = []
    for video, clip in rows:
        indices = clip["sampled_frame_indices"]
        timestamps = clip["original_frame_times"]
        frames = read_sampled_frames(video, indices)
        openings = [estimator.observe(frame)["gripper_opening"] for frame in frames]
        grays = [wrist_grayscale(frame) for frame in frames]
        intervals = []
        for slot, (before, after) in enumerate(zip(grays, grays[1:])):
            first, last = openings[slot:slot + 2]
            result = (wrist_motion_cue(before, after) if first == last == "closed"
                      else {"cue": None, "reason": "closed_jaws_not_observed_in_both_frames"})
            intervals.append({
                "first_frame": indices[slot], "last_frame": indices[slot + 1],
                "first_time": timestamps[slot], "last_time": timestamps[slot + 1],
                **result,
            })
        observations.append({
            "clip_id": clip["clip_id"], "recording_id": clip["recording_id"],
            "segment_id": clip["segment_id"], "camera": "hand",
            "intervals": intervals,
        })
    return {
        "method": "wrist_rgb_relative_motion_v1",
        "source_plan_sha256": file_digest(plan_path),
        "template": {"recording_id": TEMPLATE_RECORDING, "frame_index": TEMPLATE_FRAME,
                     "video_sha256": file_digest(template_video)},
        "observations": observations,
    }
