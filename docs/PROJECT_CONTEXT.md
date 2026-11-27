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
- To populate detections later, enable `USE_YOLO` in `main.py` and use a fresh `--output-dir`; update notebook `OUTPUT_DIR` accordingly. Existing outputs are not enriched automatically. The baseline remains independent of YOLO and no detection inference was run in this update.
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
- The live evidence, retained in `outputs/knowrob_transition_evidence_guided.json`, was used
  with the default 0.02 bonus on the 2,895 validation windows and ten seeds.
  The visual and KnowRob runs gave 33.91% vs 34.07% Accuracy, 29.95% vs
  25.16% Macro-F1, and 80.42% vs 81.56% Top-3 Accuracy. PLACE F1 remained
  near 16.33%. There were 3,130 corrections and 3,084 regressions across
  the seeds. See `outputs/experiments/reassemble_10shot_knowrob_validation/summary.json`.

## Supervised RGB fine-tuning — 2026-11-21

- `train-visual` reuses the recording split from an existing experiment plan
  and takes one central eligible window per sampled segment. It reads the
  original sampled RGB frames from the saved videos. The layers before
  `layer4` of the Kinetics-400 R(2+1)D backbone stay frozen. Training updates `layer4` and a
  new classification head, and saves the checkpoint with the best validation
  Macro-F1. Test recordings remain held out.
- The previous pilot used 32 training and 16 validation segments per class,
  for 128 training and 64 validation clips. It used seed 0, batch size 4 and
  three epochs on an NVIDIA GeForce GTX 1660 SUPER. About 65 seconds elapsed
  between saving the configuration and the final artifacts, after initial
  setup. Epoch 2 gave 59.38% accuracy and 57.54% Macro-F1. The frozen
  Prototype Classifier on the same clips gave 40.63% and 36.48%, respectively.
  The pilot used one seed and a small balanced validation sample. Its outputs
  are in `outputs/experiments/reassemble_rgb_finetune_pilot/`.
- The restored trainer passed all 20 unit tests and a one-epoch CUDA run on
  eight real training clips and eight real validation clips. The temporary
  run verified checkpoint and prediction output.
- `--train-shots all --validation-shots all` selects every eligible labeled window.
  A check against the saved REASSEMBLE plan verified 12,529 training windows
  across 81 recordings and 2,895 validation windows across 20 recordings.
  The 36 retained official test recordings are held out. All 20 unit tests
  passed after adding the full-window selection.

See `docs/FEW_SHOT_BASELINE.md` for the PowerShell command and output files.

## Training progress reporting — 2026-11-23

- `train-visual` reports input preparation, device selection, training and
  validation batch progress, and checkpoint saving. Batch progress shows
  elapsed and estimated remaining time for the current phase.
- All 20 unit tests passed. A temporary CLI smoke test used real saved
  clips and a small stand-in model on CPU to verify the progress messages
  and output files.

## Full-window RGB fine-tuning results — 2026-11-25

- Verified the completed CUDA run in
  `outputs/experiments/reassemble_rgb_finetune_full/` with 12,529 training
  and 2,895 validation windows. The run used seed 0, batch size 4 and three
  epochs. AdamW used learning rates of 0.0001 for `layer4` and 0.001 for
  the new classification head.
- Epoch 1 had the best validation Macro-F1 (68.86%), with 70.67% accuracy
  and 96.27% top-3 accuracy. The frozen Prototype Classifier evaluated on
  the same train and validation windows gave 39.47%, 45.04%, and 86.42%,
  respectively. Metrics were recomputed from the saved fine-tuned predictions.
- INSERT and REMOVE accounted for 541 of the 849 top-1 errors in epoch 1.
  The results use one seed and window-level validation. This run did not
  evaluate the held-out test recordings.
- The `reassemble_rgb_finetune_full_v2` comparison used the identical data,
  seed and training settings with both learning rates halved. Its best
  checkpoint (epoch 3) gave 69.65% Macro-F1, 54.10% REMOVE F1 and 70.33%
  accuracy. The original checkpoint gave 68.86%, 59.12% and 70.67%,
  respectively. The validation predictions reproduce both reports.
- Use epoch 1 of `reassemble_rgb_finetune_full` as the visual baseline for
  the next KnowRob experiment. The lower learning rates did not improve
  Macro-F1 and REMOVE F1 together. The official test remains held out.

## Fine-tuned predictions and KnowRob trace

- `trace-knowrob` connects selected saved predictions from the fine-tuned
  model to the existing KnowRob evidence JSON. It checks the source plan
  and recording metadata before selecting the closest non-overlapping
  predecessor. It uses the saved visual prediction of that predecessor.
- Verified the command on three real validation windows using the existing
  live KnowRob export. Both positive transitions applied once with the
  demonstration bonus of 0.02. One window had no matching transition.
  All three top-1 predictions stayed the same. All 20 unit tests passed.

## Visual predictions for missing validation context

- The 2,895 validation windows have 1,814 predecessors with saved predictions,
  1,079 distinct predecessors without predictions, and two windows with no
  predecessor. Predecessors are selected within each recording by frame
  bounds, so they do not overlap the evaluated window.
- `predict-context` generated all 1,079 missing predictions on CUDA using the
  epoch 1 checkpoint. It preserves the original clip metadata and records
  fingerprints of the checkpoint, source plan and baseline predictions.
  The output is `outputs/experiments/reassemble_rgb_finetune_full/context_predictions.json`.
  The output IDs, metadata, four-way score rankings and coverage were verified
  against the full validation timelines. All 20 unit tests passed.

## KnowRob rankings for full fine-tuned validation

- `trace-knowrob --context-file` joins the saved visual predictions with the
  additional predecessor predictions and applies the exported transitions.
  Input fingerprints tie the context predictions to the original run.
- Verified 2,895 before-and-after rankings in
  `outputs/experiments/reassemble_rgb_finetune_full/knowrob_rankings.json`.
  The 0.02 bonus applied to 431 windows for PICK to INSERT and 978 windows
  for REMOVE to PLACE. Two windows had no predecessor. All 20 unit tests passed.

## Static KnowRob transitions on full fine-tuned validation

- `compare-knowrob` compares both rankings on the same windows. The output
  is `outputs/experiments/reassemble_rgb_finetune_full/knowrob_metrics.json`.
  The visual metrics match the epoch 1 training summary.
- With a 0.02 score bonus, visual vs. KnowRob results were 70.67% vs. 70.64%
  accuracy, 68.86% vs. 68.83% Macro-F1, 76.58% vs. 76.56% INSERT F1 and
  59.12% vs. 59.03% REMOVE F1. There were zero corrections and one regression
  in 2,895 windows. The PICK to INSERT transition changed a correct REMOVE
  to INSERT in clip `2025-01-09-14-40-21_hama1_clip_0095`.
- Keep epoch 1 visual-only as the reference. This comparison covers the two
  exported static transitions, a single checkpoint and the validation split.
  All 20 unit tests and a check of the change categories passed.

## Qualitative inspection of INSERT and REMOVE — 2026-11-26

- Reviewed ten selected validation clips with INSERT and REMOVE as the top
  two visual candidates, including the KnowRob regression and controls with
  correct visual predictions. Images in `outputs/inspection_insert_remove/`
  show the saved 16 samples, additional frames around the window, and
  selected 112-by-112 views reconstructed from encoder preprocessing.
  `selection.json` preserves the clip IDs, timestamps and predictions.
- In clips 0095, 0140 and 0036, the selected three-second window shows
  approach or contact. A visible lift or extraction appears after the
  window, in the frames shown at about 5.8, 6.4 and 3.6 seconds from its
  start, respectively. The annotation continues past the window in each case.
- In clip 0123 the Ethernet interaction is near the edge of the encoder
  crop. In clip 0056 the D-SUB insertion attempt is annotated as failed.
  The inspected frames show contact with little visible state change.
  The selected sample illustrates timing and visibility issues and does
  not estimate their prevalence in validation.

## Workspace cleanup

- Kept the active 10-shot experiment plan and validation inputs, the full
  fine-tuning checkpoint and reports, the visual extraction, and the
  canonical KnowRob export. Removed duplicate exports and traces, local
  cache files, and the two standalone download and detection notebooks.
- Removed the pilot and v2 weight files and the per-seed details of the
  historical 5-shot and 10-shot KnowRob experiments. Retained their
  configurations and metric summaries, including the first 5-shot test
  summary.

## Wrist-camera RGB preparation and supervised training

- `prepare-rgb --camera hand` extracts the wrist camera and its timestamps
  from the original HDF5 files using the split and sampling settings of
  the saved REASSEMBLE experiment. It writes the MP4 files, per-recording
  metadata and a training plan with labeled window counts. Label alignment
  uses the timestamps of the selected camera.
- `train-visual --skip-prototype` consumes the RGB plan and trains from the
  saved frame indices and labels. The optional frozen prototype comparison
  remains available for plans with embeddings.
- All 21 unit tests passed. A temporary smoke test used four real REASSEMBLE
  recordings and one CUDA epoch on four training and four validation windows.
  It produced the checkpoint and validation predictions without embeddings.

See `docs/FEW_SHOT_BASELINE.md` for both PowerShell commands.

## Full wrist-camera training results — 2026-11-26

- The completed `reassemble_hand_finetune` run used 11,310 training and 2,589
  validation windows. The `hand` and `hama1` runs used identical recording
  assignments, seed, optimizer, and nominal clip sampling settings. The
  median observed clip durations were 3.24 s for `hand` and 3.00 s for `hama1`.
- The selected `hand` checkpoint is epoch 3. On each camera's full validation
  set, `hand` vs. `hama1` gave 96.16% vs. 68.86% Macro-F1, 95.87% vs. 70.67%
  accuracy, 96.70% vs. 76.58% INSERT F1, and 92.12% vs. 59.12% REMOVE F1.
- In 1,304 one-to-one window pairs from the same recording and annotated
  segment with at most 0.5 s difference at each endpoint, Macro-F1 was
  96.16% for `hand` and 72.33% for `hama1`. INSERT/REMOVE confusions totaled
  50 vs. 232. These are validation results for the two trained models.

## Static KnowRob transitions on wrist-camera validation

- Ran `predict-context`, `trace-knowrob` and `compare-knowrob` on the epoch 3
  wrist-camera checkpoint and the same static evidence export used for
  the `hama1` comparison. Inferred 1,039 missing predecessor predictions.
- The 0.02 bonus applied to 1,025 of the 2,589 validation windows: 534 for
  PICK to INSERT and 491 for REMOVE to PLACE. The full candidate order
  changed in 651 windows. The top-1 prediction stayed the same in every
  window, with zero corrections and zero regressions.
- Both rankings gave 95.87% accuracy, 96.16% Macro-F1, 96.70% INSERT F1,
  and 92.12% REMOVE F1. The paired report is
  `outputs/experiments/reassemble_hand_finetune/knowrob_metrics.json`.

- A score-gap audit of the 107 validation errors found that the two rules
  support the true label in only four cases: one INSERT predicted as PICK
  and three PLACE predicted as INSERT. The current rules cannot correct any
  of the 77 INSERT/REMOVE confusions. The full error list is in
  `outputs/experiments/reassemble_hand_finetune/hand_validation_errors.csv`.
- An exploratory replay on the same validation scores gave one correction
  and zero regressions at a bonus of 0.45 (96.20% Macro-F1). At 0.50,
  it gave one correction and one regression (96.05% Macro-F1). This is
  a sensitivity check on saved validation scores, with no bonus selected
  from the sweep for held-out testing.

## BNC state observability on wrist-camera validation

- Reviewed eight windows from eight annotated BNC segments with the epoch 3
  hand model. The sample includes two correct controls, four confusions on
  successful attempts and two confusions on unsuccessful attempts.
- In the four successful confusions, the decisive release or visible
  separation appeared in the context after the model window. Clip 0058, for
  example, showed the gripped BNC clearly separated from the fixed socket
  about 0.7 s after the end of the window.
- Aligned end-effector pose, finger positions, measured force and measured
  torque with the original hand timestamps. Jaw opening with the piece left
  behind and withdrawal with the jaws closed supported different readings.
  Rotation and torque magnitude were present in both directions, including
  unsuccessful attempts. The small selected sample does not establish a
  state detector or measure a KnowRob gain.
- The reproducible review and manual observations are in
  `notebooks/bnc_state_inspection.ipynb`. The test split was held out. The
  review used extra frames only to inspect events beyond the model windows.

## BNC socket compatibility in KnowRob — 2026-11-27

- The REASSEMBLE ontology declares `reassemble:bnc` as a connector and
  `reassemble:bncSocket` as a socket, linked by `reassemble:compatibleSocket`
  from connector to socket.
- Verified with the KnowRob Python binding in the local `knowrob/ros1:latest`
  container and the Prolog RDF backend. With `knowledge/reassemble_actions.ttl`
  loaded, the query `reassemble:compatibleSocket(reassemble:bnc, Socket)`
  returned `reassemble:bncSocket` from `answer.substitution().get("Socket")`.

## SOMA-aligned BNC scene state — 2026-11-27

- Added `knowledge/reassemble_bnc_state.ttl`, loaded with the application
  ontology, SOMA and DUL in a single KnowRob knowledge base. A timestamped
  DUL situation references the connector, socket and gripper. SOMA component
  relations connect the socket to the workstation and the gripper to the arm.
- Observations keep engagement, grip status and gripper opening independent.
  A missing status represents an unknown observation. Snapshots can carry
  the original recording, segment and clip IDs and camera timestamp.
- The live Docker smoke check retrieved the five scene entities and verified
  the change from held to unheld with the BNC still seated on two synthetic
  snapshots. On macOS, all 21 unit tests passed with `TMPDIR=/private/tmp`.
  See `docs/KNOWROB_BASELINE.md` for the live check.

## Manual RGB BNC snapshots — 2026-11-27

- Annotated the first and last sampled RGB frames of the eight validation
  clips in the BNC notebook. The 16 in-window snapshots and five retrospective
  snapshots are in `knowledge/observations/`. Each stores the source IDs,
  frame index and exact wrist-camera timestamp.
- At the sampled window endpoints, engagement is clear at the end of clip
  0021, where the connector stays seated after release. The other 15 endpoints
  omit engagement because the view is ambiguous. Four successful confusions
  have clear later seating or separation. Clip 0170 also has a later view
  with open jaws and the connector apparently seated.
- The annotations use RGB only. The later frames and the manual labels are
  reference data for inspection. Verified source IDs, partition membership,
  frame indices and timestamps for all 21 snapshots. Live KnowRob queries
  returned seated at the end of 0021 and separated after the window of 0058.

## Historical BNC action evidence pilot — 2026-11-27

- The former BNC-specific predicate mapped `separated → seated` to
  INSERT evidence and `seated → separated` to REMOVE evidence. The caller
  supplies ordered snapshots from the same evaluated window. The rule
  checks the timestamps, provenance and compatible connector–socket pair.
- Equal, missing or conflicting engagement values produce no action
  evidence. An attempted action can leave the observed state unchanged.
  The rule uses neither activity labels nor success annotations.
- The live KnowRob smoke check returned both actions on synthetic pairs and
  abstained for unchanged state, reversed times, different clips and
  conflicting observations. It abstained on all eight manual in-window
  pairs because none had two explicit engagement states. The retrospective
  observations were excluded from the check. All 21 Python unit tests passed
  with `TMPDIR=/private/tmp`.

## Automatic wrist RGB observations — 2026-11-27

- Added `observe-gripper` to estimate `open` or `closed` from the
  relative positions of the jaws, with abstention for weak matches and
  intermediate openings. The fixed-camera template and thresholds were
  selected on four support recordings. Inference uses the saved first and
  last sampled frame metadata and wrist RGB. The manual reference is read
  only during comparison, in `notebooks/bnc_rgb_state_validation.ipynb`.
- On the eight selected validation clips, all 16 opening estimates agreed
  with the manual reference (15 closed, one open), including the opening
  during 0021. All source IDs, frame indices and original timestamps
  matched the reference. This is a targeted sample, with the test split
  held out. The automatic output covers jaw opening. Grip status and
  connector engagement remain undetermined by the estimator, so these
  observations alone cannot supply evidence to the engagement rule.

## Paired KnowRob validation and state coverage — 2026-11-27

- Compared the epoch 3 visual and static KnowRob rankings on the same
  2,589 wrist-camera validation windows from 20 recordings. Both gave
  95.87% Accuracy, 96.16% Macro-F1 and 100% Top-3 Accuracy. The 0.02 static
  bonus applied to 1,025 windows and changed the candidate order on 651,
  with zero top-1 changes. On the 292 annotated BNC windows, both gave
  90.41% Accuracy and 93.01% Macro-F1, with 26 INSERT/REMOVE confusions.
- Ran the RGB opening estimator on all 5,178 saved endpoints and verified
  matching clip IDs, frame indices and original timestamps. It assigned
  opening on 3,745 endpoints (72.33%) and at both ends of 1,612 windows.
  These counts measure coverage. The engagement rule had zero eligible
  pairs, so the state arm made no queries and kept every visual ranking.
- The state audit used only the automatic RGB output. The BNC annotation
  was used for the evaluation slice. The test split stayed held out.
  The report is `outputs/experiments/reassemble_hand_finetune/knowrob_state_audit.json`.
  All 22 Python unit tests passed with `TMPDIR=/private/tmp`. See
  `docs/KNOWROB_BASELINE.md` for reproducibility.

## Proposed shared action and state model — 2026-11-27

The BNC-specific action predicate was removed. Its scene facts and manual
observations provide examples for the shared design. The two static
`preferredSuccessor` relations still serve the static baseline. The live
KnowRob check still retrieves the scene and the recorded engagement of
clip 0021. All 22 Python unit tests passed.

In the proposed model, each timestamped observation identifies the same
physical piece across frames and records whether the gripper is empty,
holding a piece, or unknown. The piece can be supported on a free surface,
seated in a specific compatible assembly site, separated from that site,
or of unknown relation to it. Gripper opening is an independent cue.
An assembly site can be a socket, hole, axle or threaded mount.

| Action | Proposed observable evidence |
| --- | --- |
| IDLE | No sustained manipulation or task-directed approach or retreat. An empty gripper can also occur during an active approach. |
| PICK | The piece leaves a free support and moves with the gripper after it is grasped. |
| PLACE | The gripper releases the piece and withdraws while the piece stays on a free support. |
| INSERT | A held piece approaches a compatible assembly site and becomes seated. The gripper can still be holding it. |
| REMOVE | A piece seated in an assembly site is grasped and leaves the site while moving with the gripper. Grasp alone is incomplete evidence. |

The transitions describe visible outcomes. Approach, alignment and contact
can support evidence of an attempt when the outcome is unchanged or
unobservable. Missing evidence leaves the action unknown. The current
four-class visual run excluded IDLE, whose REASSEMBLE annotation is
`No action.`. Check annotated idle windows before adding a fifth class.

1. Review before, during and after examples across the electrical
   connectors, pegs, gears and threaded piece, including failed attempts
   and ambiguous windows. Keep post-window evidence for manual review.
2. Validate the shared observation vocabulary and action evidence on
   manual examples, then implement rules parameterized by piece and site.
3. Check RGB observability and abstention on windows from separate validation
   recordings. Consider a detector for piece and site identity later.

### Manual review across object families

- `notebooks/reassemble_shared_state_review.ipynb` reviews seven wrist-camera
  clips from three support recordings. It uses original timestamps and
  identifies model samples, other in-window frames and retrospective context.
- The small gear shows acquisition from a free surface in 0088 and release
  in 0092. USB 0048 shows the piece staying at the receptacle after release
  in context. For removal, the same criterion applies to BNC 0138 and
  round peg 4 in 0042. The BNC is visibly separate in a later frame,
  while full separation of the peg remains uncertain in the reviewed frames.
- The threaded-piece example 0395 has an annotated failed INSERT and an
  uncertain visible outcome. The IDLE example 0001 shows empty jaws while
  the camera view moves across the board. All notebook code cells ran on
  the saved RGB and timestamp files.

### Shared catalog and observation schema

- `knowledge/reassemble_manipulation_state.ttl` defines the common snapshot,
  with separate relations to the gripper, free support and assembly site.
  Missing observations remain unknown. `knowledge/reassemble_catalog.ttl`
  maps all 17 exact object labels to their intended board sites.
- The BNC pilot maps onto the shared vocabulary through subclass and
  subproperty declarations. Its original observation IDs and timestamps
  remain intact. The existing socket compatibility is available through
  the generic `compatibleSite` query.
- Verified all 17 label strings against the support metadata. The live
  KnowRob check returned all 17 expected sites and read a real BNC snapshot
  and a synthetic peg snapshot through the same properties. It also checked
  a synthetic gear on a free support and the original BNC queries.
  All 22 Python unit tests passed.

### Generic action evidence

- `knowledge/manipulation_evidence.pl` compares two observations and returns
  the supported action, evidence kind and reason. PICK and PLACE use
  changes in grip and free support. INSERT and REMOVE use grip and
  engagement with a compatible site. BNC and pegs use the same predicate.
- Added interval cues to the shared schema. Explicit inactivity can support
  IDLE, and observed directed contact or pulling can support an INSERT or
  REMOVE attempt without a confirmed outcome. Conflicting or incomplete
  observations can yield abstention. Opening alone supplies no action evidence.
- The live KnowRob check returned all five actions and both attempt types on
  synthetic examples. It returned identical REMOVE evidence for BNC and
  round peg 4. The real BNC clip 0021 still lacks an initial engagement
  observation and returned no action evidence. Reversed times, different
  clips, incompatible sites and contradictory states also yielded
  abstention. All 22 Python unit tests passed.
- Next, measure which of the required observations can be obtained from
  RGB within the evaluated window before integrating rule evidence into
  ranking. The visual experiment still has four classes.

### Grip cue from RGB within the model window

- `observe-gripper --motion-cues` adds a separate exploratory report for
  adjacent saved wrist-camera samples. It uses closed-jaw estimates and
  compares central texture with surrounding scene motion. It makes no
  object, action or placement inference.
- Reviewed 14 validation clips covering PICK, PLACE, INSERT and REMOVE.
  The cue appeared on 22 of 210 sample pairs and in 7 of 14 clips. It
  covered 3 of the 20 manually held endpoints and none of the six unheld
  endpoints. Two additional endpoints had uncertain manual grip. Lack of a
  cue leaves grip unknown. These counts come from a selected sample.
- Verified each frame index and original timestamp against the frozen
  samples in `notebooks/rgb_grip_motion_validation.ipynb`. Support and
  engagement still need independent visual observations. The prototype
  remains separate from the action rules and reranking.
