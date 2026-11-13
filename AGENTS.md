# Repository instructions

This repository implements a master's thesis on knowledge-based robotic activity recognition.

- Read `docs/PROJECT_CONTEXT.md` before starting a new task. Consult `docs/Pipeline.pdf` for the proposed architecture if needed.
- Inspect the actual repository and Git diff before editing; distinguish implemented, tested and proposed work.
- Use Python 3.12 and the existing `.venv`. Install dependencies with `python -m pip`. Keep changes modular and avoid unnecessary refactoring or new dependencies.
- Use notebooks for exploration and visual checks; put reusable logic in `src/` and orchestration in `main.py`.
- Preserve recording IDs, segment IDs, original timestamps, labels and embedding provenance. Keep support, validation and test recordings disjoint; avoid overlapping-window leakage.
- Keep YOLO, the video encoder, the classifier, KnowRob and LLM reasoning independent for ablation experiments. Do not introduce KG or LLM changes while working on the visual baseline unless requested.
- Do not commit large datasets, generated embeddings, model weights, virtual environments or secrets. Do not overwrite or delete existing outputs without asking.
- Make small, testable changes. Explain what changed and how to run it; report tests actually run and blockers. Update `docs/PROJECT_CONTEXT.md` when a milestone is verified or an architectural decision changes. Do not commit or push unless requested.
