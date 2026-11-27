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
python3 -m src.knowrob_queries --output outputs/knowrob_transition_evidence_guided.json
```

The exporter loads the Turtle ontology into KnowRob, queries all ordered pairs
of actions and records positive answers with the ontology fingerprint. The
initial ontology is expected to yield two positive answers. The exporter
reports an error when it cannot import KnowRob or receives no positive answers.
Only the evidence JSON is needed back in the Windows workspace.
Use the existing `outputs/knowrob_transition_evidence_guided.json` or choose
a new output filename to export again.

On Linux, the exporter loads the binding with global symbol visibility so the
embedded SWI-Prolog runtime can resolve symbols when loading the RDF and socket
extensions. With the SWI-Prolog 9.0.4 build on Ubuntu 24.04, KnowRob can still
log a `plunit` domain error for `output(always)` at startup. A successful
export finishes with `Saved 2 KnowRob answers` and exit code 0.

## Compare on validation

```powershell
.\.venv\Scripts\python.exe main.py evaluate-knowrob `
  --experiment-dir outputs/experiments/reassemble_10shot_raw_10seeds `
  --evidence-file outputs/knowrob_transition_evidence_guided.json `
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

The historical KnowRob summary is retained. To reproduce its per-seed
outputs, use the saved 10-shot validation inputs and a new KnowRob output
directory.

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
  --evidence-file outputs/knowrob_transition_evidence_guided.json `
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

## Query the shared scene state and catalog

Load `DUL.owl`, `SOMA.owl`, `knowledge/reassemble_actions.ttl`,
`knowledge/reassemble_manipulation_state.ttl` and
`knowledge/reassemble_catalog.ttl` into the same KnowRob knowledge base.
For the BNC pilot data, also load `knowledge/reassemble_bnc_state.ttl`.

`compatibleSite(Piece, Site)` relates each of the 17 annotated pieces to
its intended assembly site on the task board. `datasetObjectLabel` preserves
the original spelling, including `bolt 4` for the M16 nut. [REASSEMBLE object inventory](https://arxiv.org/html/2502.05086#A3)

```prolog
reassemble:compatibleSite(reassemble:bnc, Site)
reassemble:compatibleSite(reassemble:roundPeg4, Site)
```

A `ManipulationSnapshot` is a DUL situation at one instant. It can reference
the observed piece, assembly site, free support and gripper. Engagement
(`seated` or `separated`), support (`onSupport` or `offSupport`), grip of
the observed piece, gripper occupancy and opening are independent.
Omit a property when its value is unknown. Retain the original recording,
segment and clip IDs, frame index and camera timestamp. The BNC pilot
properties map to the shared properties, and the original pilot queries
work with both ontologies loaded.

Run the live check from the repository root with the local KnowRob image:

```sh
docker run --rm --network none \
  --mount "type=bind,source=$PWD,target=/repo,readonly" \
  --entrypoint /bin/bash knowrob/ros1:latest -lc \
  'source /catkin_ws/devel/setup.bash && python3 -B /repo/tests/smoke_bnc_knowrob.py /catkin_ws/src/knowrob/owl/external'
```

The check queries all 17 intended sites. It retrieves BNC and round peg
engagement through the same snapshot properties and reads a synthetic
gear observation on a free surface. It also checks the original BNC
scene and observation queries.

For manual review of real clips, add
`knowledge/observations/bnc_validation_window.ttl` as a Turtle data source
and register the `observation` prefix with URI
`https://example.org/reassemble/observations/bnc`. For example, this query
returns `reassemble:seated`:

```prolog
reassemble:socketEngagement(observation:'2025-01-09-14-57-02_hand_clip_0021_last', Status)
```

`bnc_validation_post_window.ttl` contains later frames for retrospective
inspection. Keep both manual observation files out of automated scoring.

### Query action evidence from shared observations

Add `knowledge/manipulation_evidence.pl` as a `prolog` import in the KnowRob
reasoner, as shown in `tests/smoke_bnc_knowrob.py`. Pass two observations
from inside the evaluated window in timestamp order. They must refer to
the same recording, clip and gripper. Manipulation evidence also requires
the same piece and the same relevant support or compatible assembly site.

```prolog
manipulation_evidence(reassemble:pegAtSite, reassemble:pegRemoved, Action, Kind, Reason)
```

This synthetic example returns `remove`, `observedOutcome` and
`separatedFromCompatibleSiteWhileHeld`. The same predicate supports
`pick` and `place` from changes in support and grip, and `insert` and
`remove` from changes in engagement and grip.

For interval-level evidence, set `intervalStart` on the later snapshot to
the exact earlier snapshot. A `noTaskInteraction` cue can support `idle`.
An `alignedContactTowardSite` or `pullWhileGrippingAtSite` cue can support
an INSERT or REMOVE attempt, respectively, when the outcome is unobserved
or unchanged. `attemptOnly` leaves the outcome undetermined. With
insufficient or conflicting evidence the predicate returns no supported
action. Gripper opening alone supplies neither grip nor inactivity.

The live check exercises the five actions, both attempt cues and
abstention. These are graph queries over supplied observations. The
RGB estimator and validation rankings currently have no integration
with the generic action predicate.

### Estimate gripper opening from wrist RGB

`observe-gripper` reads the first and last of the saved sampled frames for
each selected wrist-camera validation clip. It locates the two jaws by
template matching and measures their horizontal separation. The template
comes from frame 6225 of support recording `2025-01-09-14-28-47`. Low
correlation, inconsistent jaw positions or intermediate gaps produce
`null` for `gripper_opening`. The output records source IDs, original
timestamps, frame indices and match diagnostics.

Run from the repository root with the wrist-camera recordings beside the
RGB plan and a fresh output path.

```sh
.venv/bin/python -B main.py observe-gripper \
  --plan-file outputs/visual_reassemble_hand/plan.json \
  --output outputs/bnc_state_inspection/rgb_gripper_validation.json \
  --clip-id \
  2025-01-09-14-57-02_hand_clip_0021 2025-01-13-15-18-12_hand_clip_0024 \
  2025-01-09-18-08-36_hand_clip_0146 2025-01-09-19-15-33_hand_clip_0285 \
  2025-01-13-11-01-25_hand_clip_0157 2025-01-14-12-05-01_hand_clip_0058 \
  2025-01-09-19-15-33_hand_clip_0262 2025-01-09-14-40-21_hand_clip_0170
```

Run `notebooks/bnc_rgb_state_validation.ipynb` to compare the saved
observations with the manual reference. If you chose a different output
path, set `report_path` in the notebook. The eight reviewed clips yielded
16 covered endpoints and 16 agreements, including 15 closed and one open.
These targeted clips cover a narrow range of jaw configurations. Grip
status and socket engagement need their own visual evidence. In this
sample, socket engagement was manually identifiable at only one endpoint.

### Check a grip cue from relative RGB motion

Use the same `observe-gripper` command with `--motion-cues` and a fresh
output path, such as
`outputs/bnc_state_inspection/rgb_grip_motion_validation.json`. Keep the
eight BNC clip IDs above and append these six validation clips:

```text
2025-01-09-14-57-02_hand_clip_0058  2025-01-09-18-44-37_hand_clip_0060
2025-01-09-18-08-36_hand_clip_0181  2025-01-09-14-40-21_hand_clip_0118
2025-01-09-18-08-36_hand_clip_0071  2025-01-09-14-40-21_hand_clip_0199
```

The option compares adjacent saved samples. When the jaws are estimated
closed in both frames, OpenCV tracks texture in the gap and in the
surrounding scene. A stable center against a coherently moving scene
produces `possible_held_piece`. With weak tracks or insufficient motion
the cue is unknown. Run `notebooks/rgb_grip_motion_validation.ipynb`
to verify provenance and check coverage against the manual endpoints.
The cue lacks piece identity, support and seating information. Treat it
as exploratory RGB evidence, separate from KnowRob scoring.

### Compare static KnowRob and audit state coverage

Use the saved wrist-camera epoch 3 validation predictions and the matching
static KnowRob trace. Generate opening observations for every saved validation
clip with the command below. Only clip IDs are passed to the estimator.

```sh
.venv/bin/python -B - <<'PY'
import json
from pathlib import Path
import subprocess
import sys

run = Path("outputs/experiments/reassemble_hand_finetune")
predictions = json.loads((run / "validation_predictions.json").read_text())
subprocess.run([
    sys.executable, "-B", "main.py", "observe-gripper",
    "--plan-file", "outputs/visual_reassemble_hand/plan.json",
    "--output", str(run / "rgb_gripper_validation_full.json"),
    "--clip-id", *(row["clip_id"] for row in predictions),
], check=True)
PY

.venv/bin/python -B main.py compare-knowrob \
  --rankings-file outputs/experiments/reassemble_hand_finetune/knowrob_rankings.json \
  --gripper-observations outputs/experiments/reassemble_hand_finetune/rgb_gripper_validation_full.json \
  --output outputs/experiments/reassemble_hand_finetune/knowrob_state_audit.json
```

Use fresh output paths for a repeat run. The comparison matches every
observation to the exact saved validation frame and timestamp. It reports
static bonus coverage, opening coverage and the visual fallback. The
opening estimator supplies no socket engagement states. The historical
state audit measured coverage with those inputs. The proposed shared action
model is described in `docs/PROJECT_CONTEXT.md`.
The report also includes a BNC-only evaluation slice selected by object
annotation after scoring.
