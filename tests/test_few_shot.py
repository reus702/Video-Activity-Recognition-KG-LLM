import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.batch import write_json
from src.evaluation import classification_metrics
from src.experiment import prepare_experiment, evaluate_experiment
from src.splitting import DEFAULT_CLASSES, create_split, select_support, validate_split


class FewShotTests(unittest.TestCase):

    def setUp(self):
        '''
        Build six synthetic recordings with four separable activity classes.
        '''
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.clips = []
        for record in range(6):
            directory = self.data / f"r{record}_hama1"
            (directory / "embeddings").mkdir(parents=True)
            clips = []
            for class_index, label in enumerate(DEFAULT_CLASSES):
                for segment in range(2):
                    for window in range(2):
                        clip_id = f"r{record}_{label}_{segment}_{window}"
                        vector = np.eye(4)[class_index]
                        np.save(directory / "embeddings" / f"{clip_id}.npy", vector)
                        clips.append({
                            "clip_id": clip_id, "recording_id": f"r{record}",
                            "segment_id": f"{class_index}_{segment}", "ground_truth": label,
                            "start_frame": window * 10, "end_frame_exclusive": window * 10 + 20,
                            "original_start_time": 1000 + window,
                            "original_end_time_exclusive": 1002 + window,
                            "object": "part", "success": False, "detections": [],
                            "embedding_path": f"embeddings/{clip_id}.npy"
                        })
            write_json(directory / "clips.json", clips)
            write_json(directory / "manifest.json", {
                "complete": True, "recording_id": f"r{record}", "camera": "hama1",
                "clip_count": len(clips), "config": {"embedding_dim": 4, "encoder": "synthetic"}
            })
            self.clips.extend(clips)

    def prepare(self, name="experiment"):
        '''
        Prepare a reproducible two-shot synthetic experiment.
        '''
        path = self.root / name
        plan = prepare_experiment(self.data, path, DEFAULT_CLASSES, 2, [0, 1])
        return path, plan

    def test_end_to_end_and_new_recordings(self):
        '''
        Verify deterministic plans, top-k, metrics, saved provenance and no overwrite.
        '''
        path, plan = self.prepare()
        _, second = self.prepare("repeat")
        self.assertEqual(plan, second)
        # New outputs after preparation must not alter the frozen experiment.
        extra = self.data / "new_recording"
        extra.mkdir()
        write_json(extra / "manifest.json", {"complete": False})
        summary = evaluate_experiment(path, top_k=1)
        self.assertEqual(summary["partition"], "validation")
        self.assertEqual(summary["metrics"]["accuracy"]["mean"], 1.)
        self.assertFalse((path / "test").exists())
        result = json.loads((path / "validation/seed_0.json").read_text())
        prediction = result["predictions"][0]
        self.assertEqual(len(prediction["top_k"]), 1)
        self.assertEqual(len(prediction["candidates"]), 4)
        self.assertEqual(result["metrics"]["top_3_accuracy"], 1.)
        self.assertIn("original_start_time", prediction)
        self.assertIn("detections", prediction)
        self.assertEqual(len(result["prototypes"]), 4)
        with self.assertRaises(FileExistsError):
            evaluate_experiment(path)
        test = evaluate_experiment(path, partition="test")
        self.assertEqual(test["metrics"]["accuracy"]["mean"], 1.)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_split_and_shot_leakage(self):
        '''
        Keep recording partitions disjoint and support segments unique per seed.
        '''
        split = create_split(self.clips, DEFAULT_CLASSES, 2)
        validate_split(split, self.clips, DEFAULT_CLASSES, 2)
        selected = select_support(self.clips, split["support"], DEFAULT_CLASSES, 2, 7)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({(c["recording_id"], c["segment_id"]) for c in selected}), 8)
        self.assertTrue(all(c["start_frame"] == 10 for c in selected))
        self.assertEqual(selected, select_support(self.clips, split["support"], DEFAULT_CLASSES, 2, 7))
        split["test"].append(split["support"][0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split(split, self.clips, DEFAULT_CLASSES, 2)
        with self.assertRaises(ValueError):
            select_support(self.clips, ["r0"], DEFAULT_CLASSES, 3, 0)

    def test_insufficient_coverage_no_outputs(self):
        '''
        Reject unsupported classes or shot counts without creating experiment files.
        '''
        for directory in sorted(self.data.iterdir())[1:]:
            manifest = json.loads((directory / "manifest.json").read_text())
            manifest["complete"] = False
            (directory / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.root / "experiment").exists())

    def test_manual_split(self):
        '''
        Support externally chosen or official recording assignments.
        '''
        split = {"support": ["r0", "r1"], "validation": ["r2", "r3"], "test": ["r4", "r5"]}
        file = self.root / "split.json"
        write_json(file, split)
        plan = prepare_experiment(self.data, self.root / "manual", DEFAULT_CLASSES,
                                  2, [0], split_file=file)
        self.assertEqual(plan["split"], split)
        self.assertEqual(plan["split_method"], "manual")

    def test_changed_inputs(self):
        '''
        Reject edited metadata or embeddings after experiment preparation.
        '''
        path, _ = self.prepare()
        embedding = next(self.data.glob("*/embeddings/*.npy"))
        np.save(embedding, np.ones(4))
        with self.assertRaisesRegex(ValueError, "changed"):
            evaluate_experiment(path)
        self.assertFalse((path / "validation").exists())

    def test_metric_values(self):
        '''
        Check known confusion counts, fixed-class Macro-F1 and top-three accuracy.
        '''
        predictions = []
        for truth, ranking in [("PICK", ["PICK", "INSERT", "PLACE", "REMOVE"]),
                               ("INSERT", ["PICK", "INSERT", "PLACE", "REMOVE"]),
                               ("REMOVE", ["REMOVE", "INSERT", "PICK", "PLACE"])]:
            predictions.append({"ground_truth": truth,
                                "candidates": [{"action": label, "score": 1.} for label in ranking]})
        result = classification_metrics(predictions, DEFAULT_CLASSES)
        self.assertAlmostEqual(result["accuracy"], 2/3)
        self.assertAlmostEqual(result["macro_f1"], 5/12)
        self.assertEqual(result["top_3_accuracy"], 1.)
        self.assertEqual(result["class_counts"]["PLACE"], 0)


if __name__ == "__main__":
    unittest.main()
