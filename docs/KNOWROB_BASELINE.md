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
