from bisect import bisect_right
import json
import math
from pathlib import Path

import numpy as np

from src.batch import write_json
from src.evaluation import aggregate_f1_by_class, aggregate_metrics, classification_metrics
from src.experiment import file_digest
from src.inspection import load_embedding, load_recordings
from src.knowrob_queries import ACTIONS
from src.prototype_classifier import PrototypeClassifier


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


def load_validation_context(plan):
    '''Load predecessors from the full timeline and keep labels for evaluation only.'''
    recordings = load_recordings(plan["data_dir"])
    validation_ids = set(plan["split"]["validation"])
    expected_queries, context, vectors, fingerprints = {}, {}, {}, {}
    found = set()
    for name, recording in recordings.items():
        record_id = recording["manifest"]["recording_id"]
        if record_id not in validation_ids:
            continue
        found.add(record_id)
        directory = recording["directory"]
        snapshot = plan["snapshot"][name]
        if (file_digest(directory / "manifest.json") != snapshot["manifest_sha256"]
                or file_digest(directory / "clips.json") != snapshot["clips_sha256"]):
            raise ValueError(f"Visual metadata changed for {name}")
        clips = recording["clips"]
        # Use the complete timeline, including IDLE and unlabeled windows.
        history = previous_clips(clips)
        for clip in clips:
            if clip["ground_truth"] not in plan["classes"]:
                continue
            clip_id = clip["clip_id"]
            expected_queries[clip_id] = (record_id, clip["ground_truth"])
            if clip_id not in history:
                continue
            previous = history[clip_id]
            previous_id = previous["clip_id"]
            if previous_id not in vectors:
                vectors[previous_id] = load_embedding(recording, previous)
                fingerprints[previous_id] = file_digest(directory / previous["embedding_path"])
                expected_hash = snapshot["embeddings"].get(previous_id)
                if expected_hash is not None and fingerprints[previous_id] != expected_hash:
                    raise ValueError(f"Visual embedding changed: {previous_id}")
            context[clip_id] = previous_id
    if found != validation_ids:
        raise ValueError("Validation recordings do not match the experiment plan")
    return expected_queries, context, vectors, fingerprints


def rerank_candidates(candidates, previous_action, transitions, bonus):
    '''Add soft evidence and keep the visual order whenever scores tie.'''
    ranked = []
    for candidate in candidates:
        query = transitions.get((previous_action, candidate["action"]))
        adjustment = bonus if query is not None else 0.0
        ranked.append({**candidate, "visual_score": candidate["score"],
                       "score": candidate["score"] + adjustment,
                       "knowledge_bonus": adjustment, "knowrob_query": query})
    return sorted(ranked, key=lambda candidate: candidate["score"], reverse=True)


def evaluate_knowrob(experiment_dir, evidence_path, output_dir, bonus=0.02):
    '''Compare KnowRob reranking with the saved visual validation predictions.'''
    if not math.isfinite(bonus) or not 0 < bonus <= 2:
        raise ValueError("Knowledge bonus must be in (0, 2], within the cosine score range")
    experiment_dir, evidence_path, output_dir = map(Path, (experiment_dir, evidence_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"KnowRob results already exist: {output_dir}")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("source") != "KnowRob" or evidence.get("schema_version") != 1:
        raise ValueError("Expected evidence from the KnowRob transition exporter")
    transitions = {}
    for row in evidence["transitions"]:
        key = (row["previous"], row["candidate"])
        if any(action not in ACTIONS for action in key) or key in transitions or not row["query"]:
            raise ValueError(f"Invalid KnowRob evidence for {key}")
        transitions[key] = row["query"]
    if not transitions:
        raise ValueError("No positive transitions in the KnowRob evidence")

    plan_path = experiment_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    classes = plan["classes"]
    if set(classes) != set(ACTIONS):
        raise ValueError("KnowRob transitions require the four REASSEMBLE action classes")
    validation = experiment_dir / "validation"
    visual_summary = json.loads((validation / "summary.json").read_text())
    plan_hash = file_digest(plan_path)
    if (not visual_summary["complete"] or visual_summary["plan_sha256"] != plan_hash
            or visual_summary["seeds"] != plan["seeds"]):
        raise ValueError("Visual validation summary does not match the experiment plan")
    expected, context, vectors, fingerprints = load_validation_context(plan)
    if len(expected) != visual_summary["query_count_per_seed"]:
        raise ValueError("Visual validation query count does not match the plan")

    results, input_hashes = [], {}
    for seed in plan["seeds"]:
        path = validation / f"seed_{seed}.json"
        source = json.loads(path.read_text())
        input_hashes[path.name] = file_digest(path)
        rows = source["predictions"]
        observed = {row["clip_id"]: (row["recording_id"], row["ground_truth"]) for row in rows}
        if source["seed"] != seed or len(observed) != len(rows) or observed != expected:
            raise ValueError(f"Visual validation predictions do not match for seed {seed}")

        classifier = PrototypeClassifier()
        # Reuse the fitted visual prototypes for context windows excluded from evaluation.
        classifier.prototypes = {label: np.asarray(vector, dtype=float)
                                 for label, vector in source["prototypes"].items()}
        previous_predictions = {clip_id: classifier.predict(vector, top_k=1)[0]
                                for clip_id, vector in vectors.items()}
        predictions = []
        changed = corrected = regressed = 0
        for row in rows:
            previous_id = context.get(row["clip_id"])
            prediction = previous_predictions.get(previous_id)
            action = prediction["action"] if prediction is not None else None
            visual = row["candidates"]
            ranked = rerank_candidates(visual, action, transitions, bonus)
            before, after = visual[0]["action"], ranked[0]["action"]
            truth = row["ground_truth"]
            changed += before != after
            corrected += before != truth and after == truth
            regressed += before == truth and after != truth
            predictions.append({**row, "visual_candidates": visual, "candidates": ranked,
                                "top_k": ranked[:visual_summary["display_top_k"]],
                                "predicted_activity": after,
                                "knowrob_context": {"previous_clip_id": previous_id,
                                                    "previous_prediction": prediction}})
        results.append({"seed": seed, "predictions": predictions,
                        "visual_metrics": classification_metrics(rows, classes),
                        "metrics": classification_metrics(predictions, classes),
                        "changes": {"changed": changed, "corrected": corrected, "regressed": regressed}})

    visual_metrics = [result["visual_metrics"] for result in results]
    metrics = [result["metrics"] for result in results]
    summary = {"partition": "validation", "shots": plan["shots"], "seeds": plan["seeds"],
               "query_count_per_seed": len(expected), "bonus": bonus,
               "visual_metrics": aggregate_metrics(visual_metrics),
               "metrics": aggregate_metrics(metrics),
               "visual_f1_by_class": aggregate_f1_by_class(visual_metrics),
               "f1_by_class": aggregate_f1_by_class(metrics),
               "changes_across_seeds": {key: sum(result["changes"][key] for result in results)
                                        for key in ("changed", "corrected", "regressed")},
               "plan_sha256": plan_hash, "visual_predictions_sha256": input_hashes,
               "context_embeddings_sha256": fingerprints, "evidence": evidence, "complete": True}
    output_dir.mkdir(parents=True)
    for result in results:
        write_json(output_dir / f"seed_{result['seed']}.json", result)
    write_json(output_dir / "summary.json", summary)
    return summary
