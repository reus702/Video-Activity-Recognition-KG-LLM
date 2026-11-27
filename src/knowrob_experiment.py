import json
import math
from pathlib import Path

import numpy as np

from src.batch import write_json
from src.evaluation import aggregate_f1_by_class, aggregate_metrics, classification_metrics
from src.experiment import file_digest
from src.inspection import load_embedding, load_recordings, previous_clips
from src.knowrob_queries import ACTIONS
from src.prototype_classifier import PrototypeClassifier


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


def load_transition_evidence(evidence_path):
    '''Read exported KnowRob answers keyed by previous and candidate action.'''
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
    return evidence, transitions


def trace_finetuned_predictions(run_dir, evidence_path, output_path, clip_ids=None, bonus=0.02,
                                 context_path=None):
    '''Apply exported KnowRob evidence to saved fine-tuned validation windows.'''
    run_dir, evidence_path, output_path = map(Path, (run_dir, evidence_path, output_path))
    if output_path.exists():
        raise FileExistsError(f"KnowRob trace already exists: {output_path}")
    if not math.isfinite(bonus) or not 0 < bonus <= 1:
        raise ValueError("Knowledge bonus must be in (0, 1] for softmax scores")

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    plan_path = Path(run["source_plan"])
    if not summary["complete"] or file_digest(plan_path) != run["source_plan_sha256"]:
        raise ValueError("Incomplete training run or modified source plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    predictions_path = run_dir / "validation_predictions.json"
    predictions = {row["clip_id"]: row
                   for row in json.loads(predictions_path.read_text(encoding="utf-8"))}
    prediction_hash = file_digest(predictions_path)
    if clip_ids is None:
        if context_path is None:
            raise ValueError("Full validation requires the saved context predictions")
        if predictions.keys() != {row["clip_id"] for row in run["validation_clips"]}:
            raise ValueError("Saved predictions do not match the run's validation clips")
        clip_ids = list(predictions)
    if not clip_ids or any(clip_id not in predictions for clip_id in clip_ids):
        raise ValueError("Provide clip IDs from the saved validation predictions")
    recordings = {predictions[clip_id]["recording_id"] for clip_id in clip_ids}
    if not recordings.issubset(plan["split"]["validation"]):
        raise ValueError("Requested clips are outside the validation recordings")
    evidence, transitions = load_transition_evidence(evidence_path)

    context_predictions = {}
    if context_path is not None:
        context_path = Path(context_path)
        context = json.loads(context_path.read_text(encoding="utf-8"))
        expected = {"schema_version": 1, "partition": "validation", "checkpoint_epoch": summary["best_epoch"],
                    "checkpoint_sha256": file_digest(run_dir / "checkpoint.pt"),
                    "source_plan_sha256": run["source_plan_sha256"],
                    "validation_predictions_sha256": prediction_hash, "classes": run["classes"]}
        if any(context.get(key) != value for key, value in expected.items()):
            raise ValueError("Context predictions do not match this training run")
        context_predictions = {row["clip_id"]: row for row in context["predictions"]}
        if (len(context_predictions) != len(context["predictions"])
                or predictions.keys() & context_predictions.keys()):
            raise ValueError("Context predictions contain duplicate clip IDs")

    histories = {}
    for name, recording in load_recordings(plan["data_dir"]).items():
        recording_id = recording["manifest"]["recording_id"]
        if recording_id not in recordings:
            continue
        directory = recording["directory"]
        snapshot = plan["snapshot"][name]
        if (file_digest(directory / "manifest.json") != snapshot["manifest_sha256"]
                or file_digest(directory / "clips.json") != snapshot["clips_sha256"]):
            raise ValueError(f"Visual metadata changed for {name}")
        histories[recording_id] = previous_clips(recording["clips"])

    results = []
    for clip_id in clip_ids:
        row = predictions[clip_id]
        previous = histories[row["recording_id"]].get(clip_id)
        previous_id = previous["clip_id"] if previous else None
        previous_row = predictions.get(previous_id, context_predictions.get(previous_id))
        if previous_row is not None and previous_row["recording_id"] != row["recording_id"]:
            raise ValueError(f"Predecessor prediction is from another recording: {previous_id}")
        if context_path is not None and previous is not None and previous_row is None:
            raise ValueError(f"Missing prediction for predecessor {previous_id}")
        # Without a predecessor prediction, no transition evidence applies.
        previous_action = previous_row["predicted_activity"] if previous_row else None
        visual = row["candidates"]
        ranked = rerank_candidates(visual, previous_action, transitions, bonus)
        results.append({**row, "previous_clip_id": previous_id, "previous_visual_action": previous_action,
                        "visual_candidates": visual, "candidates": ranked,
                        "visual_top1": visual[0]["action"], "predicted_activity": ranked[0]["action"]})

    report = {"partition": "validation", "classes": run["classes"],
              "checkpoint_epoch": summary["best_epoch"], "bonus": bonus,
              "prediction_file": str(predictions_path), "prediction_sha256": prediction_hash,
              "evidence": evidence, "predictions": results}
    if context_path is not None:
        report["context_file"] = str(context_path)
        report["context_sha256"] = file_digest(context_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    return report


def audit_bnc_state_coverage(observations_path, rows, visual_metrics):
    '''Report RGB opening coverage and the fallback with engagement unavailable.'''
    observations_path = Path(observations_path)
    report = json.loads(observations_path.read_text(encoding="utf-8"))
    if report.get("method") != "wrist_rgb_jaw_template_matching_v1":
        raise ValueError("Expected automatic RGB observations from observe-gripper")
    expected = {}
    for row in rows:
        if row["camera"] != "hand":
            raise ValueError("Expected wrist-camera rankings for the state audit")
        for endpoint, slot in (("first", 0), ("last", -1)):
            expected[row["clip_id"], endpoint] = (
                row["recording_id"], row["segment_id"], row["camera"],
                row["sampled_frame_indices"][slot], row["original_frame_times"][slot],
            )

    openings = {}
    for observation in report["observations"]:
        key = (observation["clip_id"], observation["endpoint"])
        provenance = tuple(observation[field] for field in
                           ("recording_id", "segment_id", "camera", "frame_index", "observed_at"))
        if key in openings or expected.get(key) != provenance:
            raise ValueError(f"RGB observation does not match its validation endpoint: {key}")
        opening = observation["gripper_opening"]
        if opening not in (None, "open", "closed") or "socket_engagement" in observation:
            raise ValueError(f"Expected an opening-only RGB observation: {key}")
        openings[key] = opening
    if set(openings) != set(expected):
        raise ValueError("RGB observations must cover the first and last endpoints of every ranked clip")

    # The RGB opening estimator supplies no engagement observations. The audit
    # reports the visual baseline as the fallback for every clip.
    return {
        "observations_file": str(observations_path),
        "observations_sha256": file_digest(observations_path),
        "source_plan_sha256": report["source_plan_sha256"],
        "observed_endpoints": len(openings),
        "endpoints_with_opening": sum(opening is not None for opening in openings.values()),
        "clips_with_two_openings": sum(
            openings[row["clip_id"], "first"] is not None
            and openings[row["clip_id"], "last"] is not None for row in rows
        ),
        "clips_with_two_engagement_states": 0,
        "rule_queries": 0,
        "abstained_clips": len(rows),
        "reason": "automatic_socket_engagement_unavailable",
        "fallback_metrics": visual_metrics,
    }


def compare_knowrob_rankings(rankings_path, output_path, *, gripper_observations=None):
    '''Compare visual and static rankings, with an optional audit of BNC state coverage.'''
    rankings_path, output_path = Path(rankings_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"KnowRob comparison already exists: {output_path}")
    report = json.loads(rankings_path.read_text(encoding="utf-8"))
    rows = report["predictions"]
    if report["partition"] != "validation" or len({row["clip_id"] for row in rows}) != len(rows):
        raise ValueError("Expected distinct validation clips in the ranking report")
    classes = report["classes"]
    visual = [{"ground_truth": row["ground_truth"], "candidates": row["visual_candidates"]} for row in rows]
    visual_metrics = classification_metrics(visual, classes)
    knowrob_metrics = classification_metrics(rows, classes)

    changes = {"changed": 0, "corrected": 0, "regressed": 0, "changed_still_wrong": 0}
    changed_clips = []
    for row in rows:
        before = row["visual_candidates"][0]["action"]
        after = row["candidates"][0]["action"]
        if before == after:
            continue
        label = row["ground_truth"]
        outcome = "changed_still_wrong"
        if after == label:
            outcome = "corrected"
        elif before == label:
            outcome = "regressed"
        changes["changed"] += 1
        changes[outcome] += 1
        changed_clips.append({"clip_id": row["clip_id"], "ground_truth": label, "visual_top1": before,
                              "knowrob_top1": after, "outcome": outcome,
                              "previous_clip_id": row["previous_clip_id"],
                              "previous_visual_action": row["previous_visual_action"],
                              "knowrob_query": row["candidates"][0]["knowrob_query"]})

    result = {"partition": report["partition"], "checkpoint_epoch": report["checkpoint_epoch"],
              "rankings_file": str(rankings_path), "rankings_sha256": file_digest(rankings_path),
              "bonus": report["bonus"], "visual_metrics": visual_metrics,
              "knowrob_metrics": knowrob_metrics, "changes": changes, "changed_clips": changed_clips}
    if gripper_observations is not None:
        result["static_coverage"] = {
            "clips_with_bonus": sum(
                any(candidate["knowledge_bonus"] > 0 for candidate in row["candidates"]) for row in rows
            ),
            "changed_candidate_order": sum(
                [candidate["action"] for candidate in row["candidates"]]
                != [candidate["action"] for candidate in row["visual_candidates"]] for row in rows
            ),
        }
        result["state_coverage"] = audit_bnc_state_coverage(gripper_observations, rows, visual_metrics)
        # Object annotations select a reporting stratum after the rankings are fixed.
        bnc_indices = [index for index, row in enumerate(rows) if row.get("object") == "BNC"]
        if bnc_indices:
            bnc_visual = classification_metrics([visual[index] for index in bnc_indices], classes)
            result["bnc_subset"] = {
                "visual_metrics": bnc_visual,
                "static_metrics": classification_metrics([rows[index] for index in bnc_indices], classes),
                "state_fallback_metrics": bnc_visual,
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def evaluate_knowrob(experiment_dir, evidence_path, output_dir, bonus=0.02):
    '''Compare KnowRob reranking with the saved visual validation predictions.'''
    if not math.isfinite(bonus) or not 0 < bonus <= 2:
        raise ValueError("Knowledge bonus must be in (0, 2], within the cosine score range")
    experiment_dir, evidence_path, output_dir = map(Path, (experiment_dir, evidence_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"KnowRob results already exist: {output_dir}")
    evidence, transitions = load_transition_evidence(evidence_path)

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
