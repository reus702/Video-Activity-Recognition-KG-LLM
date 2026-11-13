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

## Latest proposed code — verify in repository

The latest assistant response supplied `src/labeling.py` (`load_annotations`, `label_clips`) and a replacement `main.py` for batch processing. It has not been confirmed that these changes were applied or run.

Proposed batch behavior: discover `.h5` files recursively under `data/raw/reassemble/`, extract MP4 to `data/videos/`, load the encoder once, process clips, save embeddings under `outputs/embeddings/`, and save per-video `outputs/<video_id>_labeled.json`. Metadata includes `recording_id`, `segment_id`, `ground_truth`, `object`, `success`, embedding path, time bounds and detections. `USE_YOLO=False` is proposed for now. The proposed script reprocesses existing recordings; add safe reuse/skip logic if needed and verify unique IDs.

## Next milestone: visual few-shot baseline

1. Verify the current source files and run a small end-to-end batch. Add checks for timestamp/frame alignment and malformed or missing annotations. Do not assume the supplied code matches the actual local repository.
2. Add several REASSEMBLE recordings, including disassembly, and inspect class coverage by recording and segment. Start with PICK, INSERT, REMOVE, PLACE. Exclude IDLE and unlabeled windows for the initial four-class experiment; document this limitation.
3. Build a support/validation/test split by recording (use official splits if suitable). Avoid support/test overlap, and avoid counting several overlapping windows from the same action as independent few-shot examples. Aim for 5 or 10 distinct examples per class across recordings/segments.
4. Implement and test `src/prototype_classifier.py`: class means, cosine ranking, top-k candidates. Use NumPy embedding files, no encoder fine-tuning initially.
5. Evaluate on separate held-out recordings. Report Accuracy, Macro-F1, Top-3 Accuracy and class counts. Select settings on validation, not test. Only then integrate KnowRob and Qwen3.

## Open issues

- Validate alignment across additional videos; matching frame count alone does not prove alignment. Record original time boundaries in metadata where possible.
- Fixed 3 s windows may span action boundaries; strict labeling excludes these and may bias evaluation. A separate continuous-video evaluation can address transitions/IDLE later.
- The pretrained video encoder was evaluated at 16 frames at 15 fps; our 16 frames over 3 s sample motion more sparsely. Compare temporal sampling on validation later.
- YOLO COCO vocabulary does not cover many robot components or assembly objects. Inspect detections before using them for KG evidence; decide on domain-specific detection later.
- KnowRob was previously explored in Docker/ROS/Prolog; actual ontology mappings, retrieval rules and KG-only scoring are not yet implemented/verified.
