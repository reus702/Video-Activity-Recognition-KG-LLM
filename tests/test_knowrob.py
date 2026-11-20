import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.batch import write_json
from src.experiment import file_digest
from src.knowrob_experiment import evaluate_knowrob
from src.knowrob_queries import ACTIONS, query_transitions


class KnowRobTests(unittest.TestCase):
    def test_export_queries_knowrob(self):
        '''Export only positive answers returned by the KnowRob binding.'''
        class PositiveAnswer:
            pass

        queries = []
        positive = "reassemble:preferredSuccessor(reassemble:pick, reassemble:insert)"

        def submit(query, context):
            queries.append(query)
            answer = PositiveAnswer() if query == positive else object()
            return SimpleNamespace(createQueue=lambda: SimpleNamespace(pop_front=lambda: answer))

        binding = SimpleNamespace(
            KnowledgeBase=lambda settings: SimpleNamespace(submitQuery=submit),
            QueryParser=SimpleNamespace(parse=lambda query: query),
            QueryContext=lambda flag: flag,
            QueryFlag=SimpleNamespace(QUERY_FLAG_ALL_SOLUTIONS=1),
            AnswerYes=PositiveAnswer,
        )
        with patch.dict(sys.modules, {"knowrob": binding}):
            evidence = query_transitions()
        self.assertEqual(len(queries), 16)
        self.assertEqual(evidence["transitions"], [
            {"previous": "PICK", "candidate": "INSERT", "query": positive}
        ])

    def test_validation_uses_unlabeled_non_overlapping_predecessor(self):
        '''Run evaluation with synthetic evidence and preserve the original predictions.'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording = root / "data" / "r1_hama1"
            recording.mkdir(parents=True)
            clips = [
                {"clip_id": "previous", "start_frame": 0, "end_frame_exclusive": 10,
                 "ground_truth": None, "embedding_path": "previous.npy"},
                {"clip_id": "overlapping", "start_frame": 5, "end_frame_exclusive": 15,
                 "ground_truth": "IDLE", "embedding_path": "overlapping.npy"},
                {"clip_id": "query", "start_frame": 10, "end_frame_exclusive": 20,
                 "ground_truth": "INSERT", "embedding_path": "query.npy"},
            ]
            write_json(recording / "clips.json", clips)
            write_json(recording / "manifest.json", {"complete": True, "recording_id": "r1",
                                                     "clip_count": 3, "config": {"embedding_dim": 4}})
            np.save(recording / "previous.npy", [1., 0., 0., 0.])
            np.save(recording / "overlapping.npy", [0., 0., 1., 0.])

            experiment = root / "visual"
            validation = experiment / "validation"
            validation.mkdir(parents=True)
            plan = experiment / "plan.json"
            write_json(plan, {"data_dir": str(recording.parent), "classes": list(ACTIONS),
                              "shots": 10, "seeds": [0], "split": {"validation": ["r1"]},
                              "snapshot": {"r1_hama1": {
                                  "manifest_sha256": file_digest(recording / "manifest.json"),
                                  "clips_sha256": file_digest(recording / "clips.json"),
                                  "embeddings": {},
                              }}})
            write_json(validation / "summary.json", {"complete": True, "seeds": [0],
                                                     "plan_sha256": file_digest(plan),
                                                     "query_count_per_seed": 1, "display_top_k": 3})
            visual = validation / "seed_0.json"
            ranking = [{"action": action, "score": score} for action, score in
                       (("REMOVE", .71), ("INSERT", .70), ("PICK", .2), ("PLACE", .1))]
            write_json(visual, {"seed": 0, "prototypes": dict(zip(ACTIONS, np.eye(4).tolist())),
                                "predictions": [{**clips[2], "recording_id": "r1",
                                                 "candidates": ranking}]})
            original_hash = file_digest(visual)
            evidence = root / "evidence.json"
            write_json(evidence, {"schema_version": 1, "source": "KnowRob", "transitions": [
                {"previous": "PICK", "candidate": "INSERT", "query": "synthetic test answer"}
            ]})

            output = root / "reranked"
            summary = evaluate_knowrob(experiment, evidence, output)
            self.assertEqual(summary["changes_across_seeds"], {"changed": 1, "corrected": 1, "regressed": 0})
            self.assertEqual(summary["visual_metrics"]["accuracy"]["mean"], 0.)
            self.assertEqual(summary["metrics"]["accuracy"]["mean"], 1.)
            prediction = json.loads((output / "seed_0.json").read_text())["predictions"][0]
            self.assertEqual(prediction["knowrob_context"]["previous_clip_id"], "previous")
            self.assertEqual(prediction["predicted_activity"], "INSERT")
            self.assertEqual(prediction["visual_candidates"], ranking)
            self.assertEqual(file_digest(visual), original_hash)
            with self.assertRaises(FileExistsError):
                evaluate_knowrob(experiment, evidence, output)


if __name__ == "__main__":
    unittest.main()
