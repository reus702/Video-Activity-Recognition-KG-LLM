# Visual few-shot baseline

This stage reads completed outputs from `outputs/visual_baseline/`. It does not
extract video, load the encoder, or use YOLO/KG/LLM features.

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
then ranks query windows by cosine similarity. Scores are not probabilities.

Read `validation/seed_0.json` (and the other seeds):

- `predictions[*].top_k`: requested candidate activities and cosine scores;
- `predictions[*].candidates`: full class ranking;
- `predictions[*].predicted_activity`: top-1 activity;
- `metrics`: Accuracy, Macro-F1, Top-3 Accuracy, counts and confusion matrix;
- `prototypes`: fitted class vectors.

`validation/summary.json` contains metric means and population standard deviations
across support seeds. Top-3 Accuracy always uses the first three candidates in the
full ranking, independently of display `--top-k`. Confusion matrix rows are true
classes and columns are predicted classes, ordered by `class_order`.

These are **window-level** results on strictly labeled windows. Query windows may
overlap; they are not independent trials. These metrics do not evaluate IDLE,
transitions or unknown-action rejection. Segment-level evaluation is not implemented.

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

## Tests

```sh
.venv/bin/python -B -m unittest discover -s tests -v
```

The experiment tests use synthetic saved embeddings, so they need no new HDF5
processing, pretrained models or downloads.
