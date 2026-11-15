# Project context — Video Activity Recognition with KG and LLM

## Goal and architecture

Master's thesis: investigate whether structured robotic knowledge (KnowRob) and LLM reasoning improve few-shot robotic activity recognition from video, compared with a visual-only baseline.

1. Video analysis: OpenCV extracts overlapping RGB windows; YOLO detects objects/components on sampled frames.
2. Few-shot recognition: frozen pretrained torchvision R(2+1)D-18 extracts a 512-dimensional embedding per clip; a prototype classifier averages labeled support embeddings per class and ranks candidates with cosine similarity.
3. Knowledge retrieval: entity linking to KnowRob, Prolog queries, and extraction of a relevant subgraph (one KG, not a second graph). Extend the ontology if necessary.
4. LLM reasoning: proposed Qwen3-8B + KG-RAG consumes visual candidates/scores, detected objects, KG evidence and previous activities. Optional temporal validation and structured JSON output with ambiguity/UNKNOWN.
5. Evaluation: visual-only vs visual + KnowRob vs visual + KnowRob + LLM; Accuracy, Macro-F1, Top-3 Accuracy. Define a KG-only reranking method before evaluating it. Compare on identical held-out data.

Source: proposed architecture in `docs/Pipeline.pdf` (copy the original PDF into the repository). REASSEMBLE first; Baxter is a potential second dataset. Additional sensor modalities are a possible future extension, not part of the current RGB baseline.

## Current environment and data

- macOS, VS Code, Python 3.12.12 in `.venv`. Use `python -m pip`; standalone pip/pip3 previously pointed to Python 3.10.
- Dependencies: torch, torchvision, opencv-python, ultralytics, h5py, numpy, pandas, scikit-learn, matplotlib, jupyterlab. NumPy 2 ABI warning previously encountered; NumPy <2 and OpenCV <4.12 were suggested as a workaround. Verify installed versions rather than assuming.
- REASSEMBLE HDF5 recordings contain encoded camera video and original frame timestamps. First camera: `hama1`. First recording: `2025-01-09-13-57-17.h5`.
- Initial parameters: 3 s windows, 1.5 s stride, 16 sampled frames per window. YOLO11n (COCO) initially sampled frames 0, 7, 15. These are provisional choices, not final experimental settings.

## Implemented and observed in chat

- `src/dataset.py`: `extract_video()` writes encoded `hama1` bytes to MP4 and saves original camera timestamps as `.npy`.
- `src/video_processing.py`: `iter_clips()` reads overlapping windows with OpenCV and samples frames in chronological order.
- `src/feature_extraction.py`: `VideoFeatureExtractor` converts BGR to RGB, uses pretrained weights' transforms, replaces the final classifier with `nn.Identity()`, freezes the encoder and returns a `(512,)` NumPy embedding per clip.
- `src/object_detection.py`: `ObjectDetector` returns labels, confidence and bounding boxes. COCO model produced implausible labels (`sink`, `cup`); domain adaptation is deferred. Detection is independent of video embedding extraction.
- Original `main.py` ran successfully on the first recording: 67 embeddings (`0000`–`0066`) and `outputs/clips.json`.
- In `notebooks/dataset_exploration.ipynb`, HDF5 `segments_info` annotations were inspected and clip labeling tested. Example raw labels: `No action.`, `Pick Ethernet.`, `Insert Ethernet.`. Fields include `start`, `end` (Unix timestamps), `text`, `success`, and low-level annotations.
- Verified labeling result for first recording: IDLE 28, INSERT 24, PICK 7, unlabeled 8 (67 total). No REMOVE or PLACE observed in this recording.
- Labeling converts relative clip times to frame indices using FPS, indexes the original camera timestamps, and assigns a label only when a window is entirely inside an annotated segment. The tested notebook checks frame count and monotonic timestamps. Keep `success=False` as a attempted action, not as a different activity class; preserve success metadata.

## Verified preparation milestone — 2026-11-14

Implemented in the repository, with tests described below:

- `main.py` discovers HDF5 recordings recursively and rejects duplicate recording stems before processing. Default input: `data/raw/reassemble/`. Only copy completed downloads there; use a staging folder for downloads in progress.
- New outputs are isolated under `outputs/visual_baseline/<recording_id>_hama1/`: extracted MP4, original timestamp `.npy`, `embeddings/`, `clips.json` and `manifest.json`. Existing legacy outputs are untouched. Embedding paths in the new JSON are relative to the recording output directory, unlike legacy paths relative to the repository.
- Existing recording directories are reused only when completion, source identity, configuration, clip count, unique IDs and embedding integrity pass checks. Partial or incompatible directories stop the batch; choose a fresh `--output-dir` to rerun without deleting outputs. No automatic migration or partial-run recovery is implemented.
- A completion manifest is written last. It records source relative path, size and modification time (not a content hash), camera, FPS, frame count, encoder weights, preprocessing, sampling and versions. A source modification during processing aborts completion.
- `extract_video()` now refuses to overwrite MP4 or timestamp files, including when called from a notebook.
- Annotation validation checks required fields, finite increasing camera timestamps, nonempty labels, valid intervals, success flags and overlapping segments. Video frame count is checked before model loading. Structural checks do not prove temporal synchronization.
- Clip metadata adds frame bounds, sampled frame indices, relative sampled times, original camera bounds and sampled timestamps, and an explicit flag for the estimated exclusive end at recording termination. Strict full-window labeling is unchanged. `success=False` remains an attempted action of the same class.
- `src/dataset_report.py` reports window counts, represented segment counts by class, and success counts per recording. Segments with no fully contained window are absent; represented segment counts are not automatically independent shots. Each successful batch writes a new timestamped `dataset_report_*.json` snapshot for the input files discovered in that invocation.
- YOLO remains disabled and is imported only if enabled. The encoder is loaded only when a new recording needs processing; weights are explicitly pinned to `KINETICS400_V1`.
- `PrototypeClassifier` exists independently of the batch. It validates inputs, replaces old prototypes on successful refit, rejects invalid/zero vectors, and ranks by cosine similarity. Current choice: mean raw support embeddings, then cosine comparison; scores are not probabilities. Alternative normalization can be compared later on validation.

Validation actually performed:

- Python 3.12 `.venv`, `python -B -m unittest discover -s tests -v`: five tests passed, covering synthetic MP4/HDF5 batch processing with a fake encoder, safe reuse, incompatible configurations, corrupt embeddings, metadata/report counts, partial outputs, duplicate IDs, invalid annotations, overwrite protection and classifier behavior.
- Real first recording: labels recomputed in memory still match saved labels exactly (IDLE 28, INSERT 24, PICK 7, unlabeled 8).
- The production batch with pretrained encoder has not been run for this milestone. Model inference and coverage of newly added recordings remain unverified. Three `.h5` paths were visible during inspection; this is not a completed-data validation.

Run from the repository root:

```sh
.venv/bin/python main.py
.venv/bin/python main.py --output-dir outputs/visual_baseline_run_002
.venv/bin/python -B -m unittest discover -s tests -v
```

The second command is an alternative fresh run directory, not a required second execution. New HDF5 files added after file discovery are picked up on the next invocation. A failed batch preserves existing files and may leave an incomplete recording folder; it does not emit a final dataset report.

## Notebook inspection update — 2026-11-15

- `dataset_exploration.ipynb` and `video_embeddings.ipynb` now read completed outputs under `outputs/visual_baseline/`, with a selectable recording ID. Neither notebook extracts video, rewrites labels or runs an encoder/detector. No references to legacy JSON, legacy embeddings or `data/videos/` remain in these two notebooks.
- `src/inspection.py` provides shared metadata/embedding loading and optional detection flattening. Embedding paths are resolved relative to each recording directory. Dataset exploration shows window/segment coverage, validates saved labels against original annotations in memory, and previews a sampled frame with any matching stored boxes. Embedding inspection validates vectors across recordings and shows exploratory neighbors, not evaluation metrics.
- Existing `detections` schema is retained: per-frame relative video `timestamp`, then `objects` with `label`, `confidence`, pixel `bbox`. Empty or missing lists are supported. Annotation `object` is kept separate from detected labels. The notebooks display the manifest's YOLO flag, so disabled detection is not mistaken for evidence of object absence.
- To populate detections later, enable `USE_YOLO` in `main.py` and use a fresh `--output-dir`; update notebook `OUTPUT_DIR` accordingly. Existing outputs are not enriched automatically. The baseline remains independent of YOLO and no detection inference was run in this update. The old `object_detection.ipynb` remains a separate legacy demo and still references `data/videos/`.
- All code cells in both updated notebooks executed successfully against the current outputs from repository-root and notebooks working directories (plots checked with a noninteractive backend). Generated output sizes and modification times stayed unchanged. Notebook schemas validated. Six unit tests passed, including optional empty/populated detections and shared loading helpers.
- The installed pandas cannot import with the current NumPy 1.26.4 (it requests NumPy >=2.0.2). These two notebooks use escaped HTML tables instead, without changing dependencies or the baseline environment.

## Next milestone: visual few-shot evaluation

1. Run the safe batch on completed new recordings and inspect class/segment coverage, including disassembly. Initial target classes: PICK, INSERT, REMOVE, PLACE; exclude IDLE and unlabeled windows from the first experiment and document this limitation.
2. Inspect temporal alignment and sample labeled clips in notebooks across recordings.
3. Implement persistent support/validation/test splits by recording (official splits if suitable), with no recording overlap. Select shots from distinct action segments; overlapping windows are not independent examples. Target 5 or 10 examples per class where data allow.
4. Add experiment orchestration for the existing prototype classifier, repeat support sampling with recorded seeds, select settings on validation and evaluate on held-out test recordings.
5. Report Accuracy, Macro-F1, Top-3 Accuracy, confusion matrix and class counts. No split or evaluation pipeline has been implemented yet. Keep KnowRob and LLM integration deferred.

## Open issues

- Validate alignment across additional videos; matching frame count alone does not prove alignment. Original time boundaries are now stored, but temporal correctness still requires inspection.
- Fixed 3 s windows may span action boundaries; strict labeling excludes these and may bias evaluation. A separate continuous-video evaluation can address transitions/IDLE later.
- The pretrained video encoder was evaluated at 16 frames at 15 fps; our 16 frames over 3 s sample motion more sparsely. Compare temporal sampling on validation later.
- YOLO COCO vocabulary does not cover many robot components or assembly objects. Inspect detections before using them for KG evidence; decide on domain-specific detection later.
- KnowRob was previously explored in Docker/ROS/Prolog; actual ontology mappings, retrieval rules and KG-only scoring are not yet implemented/verified.
