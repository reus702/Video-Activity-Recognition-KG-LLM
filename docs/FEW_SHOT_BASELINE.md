# Visual few-shot baseline

The experiment stage reads completed clip metadata and embeddings. Prototype
classification uses the visual vectors and labeled support clips.

## Prepare REASSEMBLE using the official split

Run these commands from the repository root in PowerShell. Point
`--dataset-root` at the directory containing REASSEMBLE's `data/` and `splits/`.

```powershell
.\.venv\Scripts\python.exe main.py prepare-reassemble --dataset-root 'E:\Reassemble Dataset' --plan-file outputs/reassemble_137_plan.json --exclude-overlaps
.\.venv\Scripts\python.exe main.py --dataset-plan outputs/reassemble_137_plan.json --output-dir outputs/visual_reassemble_137
.\.venv\Scripts\python.exe main.py prepare-experiment --dataset-plan outputs/reassemble_137_plan.json --data-dir outputs/visual_reassemble_137 --experiment-dir outputs/experiments/reassemble_137_5shot --shots 5 --seeds 0 1 2
```

`prepare-reassemble` reads `train_split1.txt` and `test_split1.txt` and checks
the camera timestamps and annotations of the listed HDF5 files. The resulting
plan records the raw data path, the selected IDs in each official partition, and
the ID, partition and reason for each exclusion. The optional
`--exclude-overlaps` flag admits a subset by recording and excluding files with
overlapping segments. The default rejects overlaps. Missing files and other
invalid annotations stop preparation. Use a new plan filename for a new audit.

The batch opens the original HDF5 files at the path stored in the plan. Give it a
dedicated output directory for this selection. `prepare-experiment` requires
completed outputs matching the selected recording IDs. It derives validation
from the official train recordings with `--split-seed` (default 42) and
`--validation-fraction` (default 0.2), uses the remaining train recordings as
support, and reserves the selected official test recordings for final evaluation.
It checks coverage on saved windows and stores the dataset plan and assignments
in the experiment's `plan.json`.

For the audited local copy, the overlap policy selects 101 official train and
36 official test recordings and records 11 exclusions. Report results on this
selection as a subset of the official split. Continue with the validation and
test commands below, using `outputs/experiments/reassemble_137_5shot` as the
experiment directory.

## 1. Prepare an experiment

After preparing enough HDF5 recordings with the existing batch:

```sh
.venv/bin/python main.py prepare-experiment \
  --experiment-dir outputs/experiments/prototype_5shot \
  --shots 5 --seeds 0 1 2
```

The default classes are PICK, INSERT, REMOVE and PLACE. IDLE and unlabeled windows
are excluded; unsuccessful attempts remain in their activity class.

Each class must occur in support, validation and test recordings. Support needs at
least five distinct annotated segments per class for a five-shot experiment.
Preparation fails clearly when coverage is insufficient; it does not weaken the
recording separation or silently reduce the number of shots.

The generated `plan.json` freezes:

- recording partitions and their class/segment counts;
- one representative window from each sampled support segment;
- exact support clip IDs for each support seed;
- encoder configuration and hashes of input metadata and embeddings.

Automatic splitting defaults to 20% validation and 20% test recordings, with the
remainder for support. A seeded coverage search chooses assignments without using
embedding similarities or prediction accuracy. It is not an official split.
Fractions can be changed via `--validation-fraction` and `--test-fraction`.
A search failure is not necessarily proof that no valid assignment exists.

For official or manually chosen assignments, use `--split-file split.json`:

```json
{
  "support": ["recording_A", "recording_B"],
  "validation": ["recording_C"],
  "test": ["recording_D"]
}
```

Use actual `recording_id` values, not camera-suffixed directory names. Assign every
input recording exactly once. Keep the recording split fixed across settings and
shot-count comparisons. Each experiment requires a new output directory.

## 2. Evaluate validation

```sh
.venv/bin/python main.py evaluate-experiment \
  --experiment-dir outputs/experiments/prototype_5shot --top-k 3
```

For each support seed the classifier averages the raw support embeddings per class,
then ranks query windows by cosine similarity. To compare normalized support,
add `--normalize-support` to `prepare-experiment`. This scales each support vector
to unit L2 norm before averaging. The plan records the choice, and evaluation
applies it to the saved embeddings in memory. Scores are not probabilities.

Read `validation/seed_0.json` (and the other seeds):

- `predictions[*].top_k`: requested candidate activities and cosine scores;
- `predictions[*].candidates`: full class ranking;
- `predictions[*].predicted_activity`: top-1 activity;
- `metrics`: Accuracy, Macro-F1, Top-3 Accuracy, counts and confusion matrix;
- `prototypes`: fitted class vectors.

`validation/summary.json` contains metric means and population standard deviations
across support seeds, including per-class F1 under `f1_by_class`. Top-3 Accuracy
always uses the first three candidates in the
full ranking, independently of display `--top-k`. Confusion matrix rows are true
classes and columns are predicted classes, ordered by `class_order`.

These are **window-level** results on strictly labeled windows. Query windows may
overlap; they are not independent trials. These metrics do not evaluate IDLE,
transitions or unknown-action rejection. Segment-level evaluation is not implemented.

### Compare shot count and support normalization

Use a new experiment directory per configuration. Compare raw 5-shot versus raw
10-shot to measure the effect of support size, and raw 5-shot versus normalized
5-shot to measure the effect of normalization. Use the same dataset plan,
split seed, validation fraction, and support seeds across configurations.
At equal shot count the same seed selects the same support clips. For each run,
evaluate validation and compare `metrics.macro_f1` and `f1_by_class.PLACE` in
`validation/summary.json`.

## 3. Evaluate test after choices are fixed

```sh
.venv/bin/python main.py evaluate-experiment \
  --experiment-dir outputs/experiments/prototype_5shot --partition test
```

This creates the analogous `test/` results. Do not choose settings based on test
scores. No automatic hyperparameter tuning is performed.

Existing results are never overwritten. Modified prepared inputs are rejected.
Newly added recordings do not silently alter an existing plan. Interrupted result
folders are not resumed automatically. Plans store an absolute input root; prepare
a new experiment after relocating the data.

## Supervised RGB fine-tuning

Run from the repository root in PowerShell with a new output directory.
The input is the visual experiment plan created by `prepare-experiment`.

```powershell
.\.venv\Scripts\python.exe main.py train-visual `
  --plan-file outputs/experiments/reassemble_10shot_raw_10seeds/plan.json `
  --output-dir outputs/experiments/reassemble_rgb_finetune_run `
  --train-shots 32 --validation-shots 16 --epochs 3 --batch-size 4 `
  --seed 0 --learning-rate 0.0001
```

Each shot is one distinct labeled segment from the specified partition. The
trainer takes one central eligible clip from each selected segment and reads
its original sampled RGB frames from the saved MP4. The plan provides the
recording split. The training seed controls segment selection, head
initialization and batch shuffling. With the same plan and validation shot
count, the validation clips stay fixed across training runs. Test recordings
are held out.

For a run on every eligible window, set `--train-shots all --validation-shots all`.
With the REASSEMBLE plan above, this uses 12,529 training windows and 2,895
validation windows from the 101 retained official training recordings.
The 36 retained official test recordings stay held out. The full-window mode
preserves the observed class frequencies, including multiple windows from a
segment. Use a new output directory for the full run.

Training starts from Kinetics-400 weights and updates the last R(2+1)D residual
block and a new action classification head. The head uses ten times the
configured learning rate. The run also scores the existing frozen embeddings
with the raw Prototype Classifier on exactly the same clips.

The terminal reports input preparation, the selected device, and the first,
every 50th, and last batch of training and validation in each epoch. Each batch
update shows the percentage, elapsed time, and estimated remaining time for
that phase. The epoch summary reports loss and validation Macro-F1.

`run.json` records the configuration, input fingerprints and selected clips.
`epoch_*.json` records loss and validation metrics. `checkpoint.pt` and
`validation_predictions.json` correspond to the epoch with the highest
validation Macro-F1. Compare both visual methods in `summary.json`.

## Train on the wrist camera with RGB and labels

Run these commands from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -u main.py prepare-rgb `
  --plan-file outputs/experiments/reassemble_10shot_raw_10seeds/plan.json `
  --camera hand --output-dir outputs/visual_reassemble_hand

.\.venv\Scripts\python.exe -u main.py train-visual `
  --plan-file outputs/visual_reassemble_hand/plan.json `
  --output-dir outputs/experiments/reassemble_hand_finetune `
  --skip-prototype --train-shots all --validation-shots all `
  --epochs 3 --batch-size 4 --seed 0 --learning-rate 0.0001
```

`prepare-rgb` reads the original HDF5 recordings through the reference
plan. It extracts the wrist camera MP4 and timestamp array, describes the
same sampling policy as the reference, and labels the windows using the
wrist camera timestamps. The output includes one `clips.json` and
`manifest.json` per recording, plus `plan.json` with the inherited split,
input fingerprints and labeled window counts per partition.

`--skip-prototype` trains directly on the sampled RGB frames and labels.
For this run, `frozen_prototype_metrics` is `null` in `summary.json`.
Both commands require new output directories. The trainer uses support
and validation recordings, with the test partition reserved for later
evaluation. For a camera comparison, evaluate matching time intervals
covered by both views.

## Tests

```sh
.venv/bin/python -B -m unittest discover -s tests -v
```

The experiment tests use synthetic saved embeddings, so they need no new HDF5
processing, pretrained models or downloads.
