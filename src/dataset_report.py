from collections import Counter


def summarize_dataset(clips):
    '''
    Count windows and represented segments by recording, class and success.
    Segments without a fully contained window are not included.
    '''
    groups = {}
    for clip in clips:
        recording = clip["recording_id"]
        group = groups.setdefault(recording, {"classes": Counter(), "segments": {},
                                              "success": Counter()})
        label = clip["ground_truth"] or "UNLABELED"
        group["classes"][label] += 1
        success = clip["success"]
        group["success"]["unknown" if success is None else str(success).lower()] += 1
        if clip["segment_id"] is not None:
            group["segments"].setdefault(label, set()).add(clip["segment_id"])
    return {
        "recording_count": len(groups), "clip_count": len(clips),
        "note": "Segment counts include only segments represented by labeled windows; not independent shots.",
        "recordings": {
            name: {"windows_by_class": dict(group["classes"]),
                   "segments_by_class": {label: len(ids) for label, ids in group["segments"].items()},
                   "windows_by_success": dict(group["success"])}
            for name, group in sorted(groups.items())
        }
    }
