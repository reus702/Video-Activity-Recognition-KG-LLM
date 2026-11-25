# KnowRob validation experiment

The first experiment tests whether weak temporal evidence can improve the
saved 10-shot ranking. The application ontology in
`knowledge/reassemble_actions.ttl` describes the four actions and two preferred
successors, PICK to INSERT and REMOVE to PLACE. These are advisory task
transitions. A robot may still be performing the same action in the next window.

## Export the query answers

Use a Linux environment with the
[KnowRob 2 Python bindings](https://github.com/knowrob/knowrob/blob/dev/src/integration/python/README.md)
and its Prolog backend available. From the repository root in that environment:

```sh
python3 -c "import knowrob"
python3 -m src.knowrob_queries --output outputs/knowrob_transition_evidence.json
```

The exporter loads the Turtle ontology into KnowRob, queries all ordered pairs
of actions and records positive answers with the ontology fingerprint. The
initial ontology is expected to yield two positive answers. The exporter
reports an error when it cannot import KnowRob or receives no positive answers.
Only the evidence JSON is needed back in the Windows workspace.

On Linux, the exporter loads the binding with global symbol visibility so the
embedded SWI-Prolog runtime can resolve symbols when loading the RDF and socket
extensions. With the SWI-Prolog 9.0.4 build on Ubuntu 24.04, KnowRob can still
log a `plunit` domain error for `output(always)` at startup. A successful
export finishes with `Saved 2 KnowRob answers` and exit code 0.

## Compare on validation

```powershell
.\.venv\Scripts\python.exe main.py evaluate-knowrob `
  --experiment-dir outputs/experiments/reassemble_10shot_raw_10seeds `
  --evidence-file outputs/knowrob_transition_evidence.json `
  --output-dir outputs/experiments/reassemble_10shot_knowrob_validation
```

For each seed and query, find the closest earlier window ending at or before
the current window starts in the same recording. The context comes from all
saved windows, including IDLE and unlabeled windows, and uses the earlier
window's visual prediction computed with the saved prototypes. The query's
ground truth is used to compute evaluation metrics.

Positive KnowRob evidence adds a provisional 0.02 to the candidate's visual
score (`--bonus`). The other candidates retain their visual scores. Each seed
result preserves both rankings and the query that supported any bonus.
`summary.json` compares visual and reranked metrics and counts changed top-1
predictions, corrections and regressions across the support seeds. Per-class
F1 is under `visual_f1_by_class` and `f1_by_class`. Use a new output directory
for each run.

## Trace the fine-tuned validation predictions

The `trace-knowrob` command applies the exported static evidence to selected
clips from a completed `train-visual` run. It uses the full clip timeline to
find the closest non-overlapping predecessor in the same recording. If the
predecessor lacks a saved visual prediction, the trace leaves temporal
context unknown. The annotation label is displayed for inspection after
reranking. The command preserves both rankings and the matching queries.

From the repository root in PowerShell, using the existing KnowRob export:

```powershell
.\.venv\Scripts\python.exe main.py trace-knowrob `
  --run-dir outputs/experiments/reassemble_rgb_finetune_full `
  --evidence-file outputs/knowrob_transition_evidence.json `
  --output outputs/experiments/reassemble_rgb_knowrob_trial.json `
  --clip-id 2025-01-09-14-40-21_hama1_clip_0004 2025-01-09-14-40-21_hama1_clip_0014 2025-01-09-14-40-21_hama1_clip_0017
```

These three illustrative clips cover both exported transitions and a missing
transition. The default bonus of 0.02 is for tracing score changes. Use a new
output path for each invocation.

## Predict the missing validation context

From the repository root in PowerShell, use a new output path:

```powershell
.\.venv\Scripts\python.exe -u main.py predict-context `
  --run-dir outputs/experiments/reassemble_rgb_finetune_full `
  --output outputs/experiments/reassemble_rgb_finetune_full/context_predictions_repeat.json
```

The command targets non-overlapping predecessors absent from
`validation_predictions.json`. It reads their sampled RGB frames, verifies
the source videos, and runs the saved classifier in inference mode.
The separate JSON records the clip IDs, original timestamps, class rankings
and input fingerprints. These predictions provide temporal context for the
later KnowRob evaluation.

## Apply the exported transitions to all fine-tuned validation clips

From the repository root in PowerShell, supply the saved context and
omit `--clip-id` to process the full validation partition:

```powershell
.\.venv\Scripts\python.exe main.py trace-knowrob `
  --run-dir outputs/experiments/reassemble_rgb_finetune_full `
  --context-file outputs/experiments/reassemble_rgb_finetune_full/context_predictions.json `
  --evidence-file outputs/knowrob_transition_evidence_guided.json `
  --output outputs/experiments/reassemble_rgb_finetune_full/knowrob_rankings_repeat.json `
  --bonus 0.02
```

The report stores the previous clip and its visual prediction, both rankings,
and the query and score adjustment for each supported candidate. The command
validates the context file against the checkpoint, source plan and saved
validation predictions before applying the bonus.

## Compare the visual and KnowRob rankings

```powershell
.\.venv\Scripts\python.exe main.py compare-knowrob `
  --rankings-file outputs/experiments/reassemble_rgb_finetune_full/knowrob_rankings.json `
  --output outputs/experiments/reassemble_rgb_finetune_full/knowrob_metrics_repeat.json
```

The JSON records the two sets of window-level metrics and lists the clips
whose top-1 label changed. A change is a correction if it matches the
annotation, a regression if the original label matched the annotation, or
a change between two incorrect labels. Use a new output path.
