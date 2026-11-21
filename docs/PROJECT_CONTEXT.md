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
3. Use the implemented persistent recording splits and segment-based support selection described below; validate coverage before preparing a 5- or 10-shot experiment. Official splits can be supplied manually.
4. Run the implemented experiment commands with recorded support seeds, choose settings on validation, then explicitly evaluate held-out test recordings.
5. Inspect saved Accuracy, Macro-F1, Top-3 Accuracy, confusion matrix and class counts. The split/evaluation pipeline is implemented and tested synthetically; real held-out evaluation remains to be run. Keep KnowRob and LLM integration deferred.

## Open issues

- Validate alignment across additional videos; matching frame count alone does not prove alignment. Original time boundaries are now stored, but temporal correctness still requires inspection.
- Fixed 3 s windows may span action boundaries; strict labeling excludes these and may bias evaluation. A separate continuous-video evaluation can address transitions/IDLE later.
- The pretrained video encoder was evaluated at 16 frames at 15 fps; our 16 frames over 3 s sample motion more sparsely. Compare temporal sampling on validation later.
- YOLO COCO vocabulary does not cover many robot components or assembly objects. Inspect detections before using them for KG evidence; decide on domain-specific detection later.
- KnowRob was previously explored in Docker/ROS/Prolog; actual ontology mappings, retrieval rules and KG-only scoring are not yet implemented/verified.


## Few-shot experiment implementation — 2026-11-15

- Added `src/splitting.py`, `src/experiment.py`, `src/evaluation.py` and `tests/test_few_shot.py`. No new HDF5 recordings were processed for this milestone; encoder, YOLO, KnowRob and LLM are not used by the experiment runner.
- `main.py prepare-experiment` freezes support/validation/test recording IDs and selected support clips in a new experiment directory. Default classes: PICK, INSERT, REMOVE, PLACE; IDLE, unlabeled windows and unrequested classes are excluded. Failed attempts remain in their activity class.
- Automatic splitting uses a seeded class-coverage search, not prediction scores or an official REASSEMBLE split. Small search spaces are exhaustive; larger searches are limited to 5000 assignments. Default fractions are 20% validation and 20% test. A manual or official assignment can be passed via `--split-file`: a JSON object containing support/validation/test lists of recording IDs. Every input recording must be assigned exactly once; multiple camera outputs for one recording are rejected.
- Every class must occur in each partition. Support must provide K distinct `(recording_id, segment_id)` pairs per class. Each sampled segment contributes its central eligible window. Multiple segments from a support recording are allowed; these are not guaranteed statistically independent examples.
- `plan.json` records class order, shots, split/support seeds, exact support clip IDs, configuration and SHA256 fingerprints of manifests, clip metadata and eligible embeddings. Mixed extraction configurations are rejected. Newly added recordings do not change an existing plan. Modified prepared inputs are rejected at evaluation. Plans store an absolute input root; prepare a new plan after relocating the dataset.
- `main.py evaluate-experiment` defaults to validation and connects the existing raw-mean/cosine Prototype Classifier to saved embeddings. Explicit `--partition test` is required for test evaluation. Select settings on validation and keep recording splits fixed across comparisons; no automatic tuning is implemented.
- Each partition saves `seed_<seed>.json` with prototypes, predictions, requested `top_k`, full `candidates`, original clip metadata and metrics, plus `summary.json` with mean/population standard deviation across seeds. These are not confidence intervals. Accuracy, fixed-class Macro-F1, Top-3 Accuracy, confusion matrix (rows=true, columns=predicted), class counts and per-class F1 are computed without new dependencies. Top-3 uses full rankings even if displayed top-k is smaller. With fewer than three classes Top-3 is trivial.
- Evaluation uses all eligible held-out windows; windows may overlap. Metrics describe strictly labeled action windows, not independent trials or continuous-video performance. Segment-level evaluation and unknown-action rejection remain future work. Optional detections are preserved as metadata but do not influence classification.
- Existing outputs are not overwritten. Interrupted experiment/result directories are not resumed automatically; preserve them and use a new experiment directory. `video_embeddings.ipynb` includes a read-only viewer for saved top-k results.
- Full usage and output descriptions are in `docs/FEW_SHOT_BASELINE.md`.

Commands after enough recordings have been processed:

```sh
.venv/bin/python main.py prepare-experiment --experiment-dir outputs/experiments/prototype_5shot --shots 5 --seeds 0 1 2
.venv/bin/python main.py evaluate-experiment --experiment-dir outputs/experiments/prototype_5shot
# Only after validation choices are fixed:
.venv/bin/python main.py evaluate-experiment --experiment-dir outputs/experiments/prototype_5shot --partition test
```

Validation: all 12 unit tests passed in Python 3.12, including the synthetic end-to-end few-shot run, manual/automatic splits, leakage and shot checks, reproducibility, changed-input rejection, safe outputs and known metric values. Both CLI help commands were checked. The new notebook viewer was validated with empty and synthetic populated results. No real held-out recognition score is claimed here. The three previously completed recordings alone lack sufficient four-class recording coverage for this protocol.


## Source timestamp portability fix — 2026-11-15

A moved recording retained identical source name, relative path, size and pipeline configuration,
but its modification time differed by 42 ns. Source reuse now compares `mtime_ns` at microsecond
precision, ignoring only digits below one microsecond. Raw source timestamps remain unchanged in
manifests. File size/path/name and full configuration checks remain strict; this stat-based identity
is still not a content checksum. A new regression test covers accepted submicrosecond rounding,
rejected microsecond changes, size/path changes and unchanged manifest metadata. No dataset output
or existing manifest is rewritten by this fix.

## Official REASSEMBLE selection — 2026-11-20

- `prepare-reassemble` creates a dataset plan from the official training and
  test lists. It checks annotation structure and camera timestamps. The explicit
  overlap exclusion option records each excluded ID, its official partition,
  and the reason. The batch reads selected HDF5 files directly from the original
  dataset directory through `--dataset-plan`.
- The local plan at `outputs/reassemble_137_plan.json` selects 101 train and
  36 test recordings. It records 11 official recordings excluded for overlaps
  (10 train and 1 test). The results from this selection describe a subset of
  the official split.
- Experiment preparation accepts the same plan and derives support and
  validation from official train. The default seed and validation fraction give
  81 support, 20 validation and 36 test recordings. A metadata check on the
  137 real recordings confirmed coverage of all four classes and at least five
  distinct support segments per class. Experiment preparation repeats this
  check on the saved labeled windows.
- All 16 unit tests passed in the Windows Python 3.12 environment using a
  test-runner workaround for sandbox permissions on temporary directories.
  A separate temporary batch opened two real HDF5 files from the external drive
  through a dataset plan and produced 73 valid pretrained R(2+1)D-18 embeddings.

## Full visual extraction for the selected REASSEMBLE recordings — 2026-11-20

- The completed batch in `outputs/visual_reassemble_137/` contains 137
  recording manifests and 28,622 clip embeddings. The final report is
  `dataset_report_20261120T135503639048Z.json`. All selected recording IDs
  have complete manifests, and every clip references an existing embedding.
- Saved metadata contains 19,907 windows in the four target classes, 163 IDLE
  windows and 8,552 unlabeled windows. The default recording split satisfies
  5-shot coverage on the saved metadata, with 2,895 validation and 4,483
  test windows in the target classes.

## First real visual 5-shot evaluation — 2026-11-20

- The current local run uses `outputs/reassemble_plan.json` and
  `outputs/visual_reassemble/`. The frozen experiment and both result
  partitions are in `outputs/experiments/reassemble_5shot/`. The experiment
  used 81 support, 20 validation and 36 test recordings, with five distinct
  support segments per class and support seeds 0, 1 and 2.
- Validation (2,895 windows) gave 34.29% Accuracy, 30.47% Macro-F1 and
  81.77% Top-3 Accuracy. Test (4,483 windows) gave 32.69% Accuracy,
  30.07% Macro-F1 and 82.76% Top-3 Accuracy. These are window-level means
  across support seeds on the selected subset of the official split.
- A constant ranking by class frequency over labeled windows in the support
  recordings gave 49.15% Accuracy, 16.48% Macro-F1 and 92.37% Top-3 Accuracy
  on validation. This is a class-imbalance diagnostic using labels from the
  full support pool. Validation F1 was lowest for PLACE (15.99%).

## Shot count and support normalization comparisons — 2026-11-20

- `prepare-experiment --normalize-support` freezes the choice to L2-normalize
  each support embedding before computing the class mean. Raw means remain the
  default. Both policies read the saved vectors and rank queries by cosine
  similarity. At equal shot count and seed they use the same support clips.
- New evaluation summaries include per-class F1 means and population standard
  deviations across seeds. Compare raw 5-shot, raw 10-shot, and normalized
  5-shot on the same validation recordings.
- All 17 unit tests passed in the Windows Python 3.12 environment using the
  temporary-directory workaround. A temporary real 10-shot normalized run
  used the saved split and 2,895 validation windows, and reported PLACE F1.

## Validation comparison across ten support seeds — 2026-11-20

All three runs used the same 2,895 validation windows, recording split, and
embedding fingerprints. The two 5-shot runs used identical support clips
per seed. Each configuration was evaluated on validation only.

| Configuration | Accuracy | Macro-F1 | PLACE F1 | Top-3 Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 5-shot, raw support | 32.86% | 29.12% | 14.87% | 80.13% |
| 10-shot, raw support | 33.91% | 29.95% | 16.34% | 80.42% |
| 5-shot, normalized support | 32.97% | 29.20% | 14.72% | 80.34% |

Raw 10-shot improved Macro-F1 by 0.83 percentage points and PLACE F1 by
1.46 points over raw 5-shot. Both metrics improved in eight of ten seed
comparisons. The support draws for different shot counts were sampled
separately. Normalizing 5-shot support changed Macro-F1 by +0.08 points and
PLACE F1 by -0.15 points. Results are under `outputs/experiments/` in
`reassemble_5shot_raw_10seeds`, `reassemble_10shot_raw_10seeds`, and
`reassemble_5shot_norm_10seeds`.

See `docs/FEW_SHOT_BASELINE.md` for the commands.

## First KnowRob integration — 2026-11-20

- The application ontology contains the four REASSEMBLE actions and two
  preferred task transitions. The KnowRob 2 exporter queries those relations
  through its Python binding and records the positive answers.
- `evaluate-knowrob` reranks the four saved visual candidates on validation
  with a soft bonus. The previous visual action comes from the closest
  non-overlapping window in the same recording. The history includes all
  windows, and query labels are used only for evaluation. Each result records
  the evidence, visual and reranked scores, corrections and regressions.
- All 19 unit tests passed. A run with synthetic evidence over the real
  10-shot outputs reproduced the original visual metrics on 2,895 windows
  for ten seeds. A live runtime was unavailable during that verification.

See `docs/KNOWROB_BASELINE.md` for the export and evaluation commands.

## Live KnowRob export and validation — 2026-11-21

- On Ubuntu 24.04 in WSL2, KnowRob commit
  `d03b865927d8634cef30e2e4ef9b60e948777b35` was built against SWI-Prolog 9.0.4.
  The Python exporter now imports the binding with global symbol visibility
  so SWI-Prolog can load its native RDF and socket modules. The exporter exited
  successfully and saved the two preferred transitions. SWI-Prolog still
  logs the upstream `plunit` diagnostic for `output(always)` at startup.
- All 19 unit tests passed, including verification that the exporter restores
  the interpreter's original loader flags after the import.
- The live evidence in `outputs/knowrob_transition_evidence.json` was used
  with the default 0.02 bonus on the 2,895 validation windows and ten seeds.
  The visual and KnowRob runs gave 33.91% vs 34.07% Accuracy, 29.95% vs
  25.16% Macro-F1, and 80.42% vs 81.56% Top-3 Accuracy. PLACE F1 remained
  near 16.33%. There were 3,130 corrections and 3,084 regressions across
  the seeds. See `outputs/experiments/reassemble_10shot_knowrob_validation/summary.json`.
