import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import h5py
import numpy as np

import main
from src.batch import load_completed
from src.dataset import extract_video
from src.inspection import detection_rows, load_embedding, load_recordings
from src.labeling import OverlappingSegmentsError, load_annotations
from src.prototype_classifier import PrototypeClassifier
from src.reassemble import load_reassemble_plan, prepare_reassemble
from src.video_processing import iter_clips, read_sampled_frames


class VisualPipelineTests(unittest.TestCase):

    def setUp(self):
        '''
        Create a small real video inside an annotated HDF5 recording.
        '''
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "data"
        self.raw.mkdir()
        video = self.root / "input.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
        if not writer.isOpened():
            self.fail("Cannot create test MP4")
        for i in range(60):
            writer.write(np.full((32, 32, 3), i, dtype=np.uint8))
        writer.release()
        self.h5 = self.raw / "recording.h5"
        with h5py.File(self.h5, "w") as f:
            f["hama1"] = np.frombuffer(video.read_bytes(), dtype=np.uint8)
            f["timestamps/hama1"] = 1000 + np.arange(60) / 10
            group = f.create_group("segments_info/0")
            for key, value in {"start": 999., "end": 1007., "text": b"Pick part.",
                               "success": False}.items():
                group[key] = value

    def run_batch(self, *input_args):
        '''
        Run the batch with a fake encoder to avoid model downloads.
        '''
        module = types.ModuleType("src.feature_extraction")
        class FakeEncoder:
            def extract(self, frames):
                '''
                Return a deterministic nonzero vector for orchestration tests.
                '''
                return np.ones(512, dtype=np.float32)
        module.VideoFeatureExtractor = FakeEncoder
        inputs = input_args or ("--raw-dir", str(self.raw))
        with patch.dict(sys.modules, {"src.feature_extraction": module}), patch.object(
                sys, "argv", ["main.py", *inputs,
                              "--output-dir", str(self.root / "output")]), contextlib.redirect_stdout(io.StringIO()):
            main.main()

    def test_batch_reuse_and_metadata(self):
        '''
        Check completion, frame provenance, report counts and safe reuse.
        '''
        self.run_batch()
        directory = self.root / "output/recording_hama1"
        clips = json.loads((directory / "clips.json").read_text())
        loaded = load_recordings(self.root / "output")["recording_hama1"]
        self.assertEqual(load_embedding(loaded, clips[0]).shape, (512,))
        self.assertEqual(len(clips), 3)
        self.assertEqual(clips[0]["ground_truth"], "PICK")
        self.assertFalse(clips[0]["success"])
        self.assertEqual(len(clips[0]["original_frame_times"]), 16)
        self.assertEqual(clips[0]["original_start_time"], 1000)
        manifest_path = directory / "manifest.json"
        modified = manifest_path.stat().st_mtime_ns
        self.run_batch()
        self.assertEqual(manifest_path.stat().st_mtime_ns, modified)
        reports = list((self.root / "output").glob("dataset_report_*.json"))
        self.assertEqual(len(reports), 2)
        report = json.loads(reports[0].read_text())
        self.assertEqual(report["recordings"]["recording"]["segments_by_class"], {"PICK": 1})
        manifest = json.loads(manifest_path.read_text())
        with self.assertRaises(ValueError):
            load_completed(directory, manifest["source"], {**manifest["config"], "num_frames": 8}, "hama1")
        np.save(directory / clips[0]["embedding_path"], np.zeros(512))
        with self.assertRaises(ValueError):
            load_completed(directory, manifest["source"], manifest["config"], "hama1")

    def test_source_timestamp_precision(self):
        '''
        Allow submicrosecond copy rounding while rejecting other source changes.
        '''
        self.run_batch()
        directory = self.root / "output/recording_hama1"
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        saved = manifest["source"]
        saved["mtime_ns"] = (saved["mtime_ns"] // 1000) * 1000 + 100
        path.write_text(json.dumps(manifest))
        copied = {**saved, "mtime_ns": saved["mtime_ns"] + 42}
        self.assertEqual(len(load_completed(directory, copied, manifest["config"], "hama1")), 3)
        for changed in ({**saved, "mtime_ns": saved["mtime_ns"] + 1000},
                        {**saved, "size": saved["size"] + 1},
                        {**saved, "relative_path": "another.h5"}):
            with self.assertRaises(ValueError):
                load_completed(directory, changed, manifest["config"], "hama1")
        self.assertEqual(json.loads(path.read_text())["source"], saved)

    def test_partial_output_and_duplicate_ids(self):
        '''
        Reject incomplete output folders and duplicate recording names.
        '''
        directory = self.root / "output/recording_hama1"
        directory.mkdir(parents=True)
        with self.assertRaises(ValueError):
            self.run_batch()
        (self.raw / "nested").mkdir()
        (self.raw / "nested/recording.h5").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_batch()

    def test_invalid_annotations(self):
        '''
        Reject nonfinite timestamps and overlapping annotations.
        '''
        with h5py.File(self.h5, "r+") as f:
            f["timestamps/hama1"][0] = np.nan
        with self.assertRaises(ValueError):
            load_annotations(self.h5)
        with h5py.File(self.h5, "r+") as f:
            f["timestamps/hama1"][0] = 1000
            f.copy("segments_info/0", "segments_info/1")
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            load_annotations(self.h5)

    def test_reassemble_plan_drives_batch(self):
        '''Process official inputs in place and record explicit overlap exclusions.'''
        splits = self.root / "splits"
        splits.mkdir()
        (splits / "train_split1.txt").write_text("overlap\nrecording\n")
        (splits / "test_split1.txt").write_text("query\n")
        shutil.copyfile(self.h5, self.raw / "query.h5")
        overlap = self.raw / "overlap.h5"
        shutil.copyfile(self.h5, overlap)
        with h5py.File(overlap, "r+") as source:
            source.copy("segments_info/0", "segments_info/1")
        (self.raw / "unlisted.h5").write_bytes(b"not an HDF5 file")

        plan_path = self.root / "plan.json"
        with self.assertRaisesRegex(OverlappingSegmentsError, "overlap.h5"):
            prepare_reassemble(self.root, plan_path)
        self.assertFalse(plan_path.exists())
        with patch.object(sys, "argv", ["main.py", "prepare-reassemble", "--dataset-root",
                                        str(self.root), "--plan-file", str(plan_path),
                                        "--exclude-overlaps"]), contextlib.redirect_stdout(io.StringIO()):
            main.main()

        plan = load_reassemble_plan(plan_path)
        self.assertEqual(plan["split"], {"train": ["recording"], "test": ["query"]})
        self.assertEqual(plan["excluded"]["overlap"], {
            "partition": "train", "reason": "Overlapping annotation segments"
        })
        with self.assertRaises(FileExistsError):
            prepare_reassemble(self.root, plan_path, exclude_overlaps=True)

        self.run_batch("--dataset-plan", str(plan_path))
        output = self.root / "output"
        self.assertEqual(set(load_recordings(output)), {"recording_hama1", "query_hama1"})
        report = json.loads(next(output.glob("dataset_report_*.json")).read_text())
        self.assertEqual(report["input_root"], str(self.raw))
        self.assertEqual(report["dataset_plan"], str(plan_path))
        (self.raw / "query.h5").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "query.h5"):
            self.run_batch("--dataset-plan", str(plan_path))

    def test_reassemble_rejects_malformed_official_inputs(self):
        '''Reject leakage and unexpected file errors even when overlaps can be excluded.'''
        splits = self.root / "splits"
        splits.mkdir()
        (splits / "train_split1.txt").write_text("recording\n")
        test = splits / "test_split1.txt"
        test.write_text("recording\n")
        plan_path = self.root / "plan.json"
        with self.assertRaisesRegex(ValueError, "Duplicate recording IDs"):
            prepare_reassemble(self.root, plan_path, exclude_overlaps=True)
        test.write_text("broken\n")
        (self.raw / "broken.h5").write_bytes(b"not an HDF5 file")
        with self.assertRaisesRegex(ValueError, "Cannot validate.*broken.h5"):
            prepare_reassemble(self.root, plan_path, exclude_overlaps=True)
        self.assertFalse(plan_path.exists())

    def test_extraction_never_overwrites(self):
        '''
        Preserve existing extracted video and timestamp files.
        '''
        video = extract_video(self.h5, self.root / "videos")
        original = video.read_bytes()
        with self.assertRaises(FileExistsError):
            extract_video(self.h5, self.root / "videos")
        self.assertEqual(video.read_bytes(), original)

    def test_training_reads_original_sampled_frames(self):
        video = self.root / "input.mp4"
        for clip in iter_clips(video):
            np.testing.assert_array_equal(
                read_sampled_frames(video, clip["sampled_frame_indices"]), clip["frames"]
            )

    def test_optional_detections(self):
        '''
        Keep empty detections valid and annotation objects separate from YOLO.
        '''
        clip = {"clip_id": "clip", "recording_id": "recording", "object": "AnnotatedPart"}
        self.assertEqual(detection_rows([clip]), [])
        clip["detections"] = []
        self.assertEqual(detection_rows([clip]), [])
        clip["detections"] = [
            {"timestamp": 0., "objects": []},
            {"timestamp": 1., "objects": [
                {"label": "DetectedPart", "confidence": 0.9, "bbox": [1, 2, 3, 4]}
            ]}
        ]
        rows = detection_rows([clip])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "DetectedPart")
        self.assertEqual(rows[0]["timestamp"], 1.)
        self.assertEqual(clip["object"], "AnnotatedPart")

    def test_classifier(self):
        '''
        Check means, ranking, refit and invalid input handling.
        '''
        classifier = PrototypeClassifier()
        with self.assertRaises(ValueError):
            classifier.predict([1., 0.])
        classifier.fit([[1., 0.], [3., 0.], [0., 1.]], ["PICK", "PICK", "INSERT"])
        np.testing.assert_array_equal(classifier.prototypes["PICK"], [2., 0.])
        self.assertEqual(classifier.predict([1., 0.])[0]["action"], "PICK")
        classifier.fit([[0., 1.]], ["PLACE"])
        self.assertEqual(set(classifier.prototypes), {"PLACE"})
        for vector in ([0., 0.], [np.nan, 1.], [1., 2., 3.]):
            with self.assertRaises(ValueError):
                classifier.predict(vector)
        with self.assertRaises(ValueError):
            classifier.fit([[0., 0.]], ["PICK"])
        with self.assertRaises(ValueError):
            classifier.predict([1., 0.], top_k=0)


if __name__ == "__main__":
    unittest.main()
