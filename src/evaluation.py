import numpy as np


def classification_metrics(predictions, classes):
    '''
    Compute window-level metrics with a fixed class order and zero F1 for absent classes.
    '''
    if not predictions:
        raise ValueError("No query predictions to evaluate")
    positions = {label: i for i, label in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    top3 = 0
    for row in predictions:
        truth = row["ground_truth"]
        ranked = row["candidates"]
        if not ranked:
            raise ValueError("Empty candidate ranking")
        confusion[positions[truth], positions[ranked[0]["action"]]] += 1
        top3 += truth in [candidate["action"] for candidate in ranked[:3]]
    true_positive = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(len(classes)), where=denominator != 0)
    return {
        "unit": "window", "query_count": len(predictions),
        "accuracy": float(true_positive.sum() / len(predictions)),
        "macro_f1": float(f1.mean()), "top_3_accuracy": float(top3 / len(predictions)),
        "class_order": list(classes), "confusion_matrix": confusion.tolist(),
        "class_counts": {label: int(confusion[i].sum()) for i, label in enumerate(classes)},
        "f1_by_class": {label: float(f1[i]) for i, label in enumerate(classes)}
    }


def aggregate_metrics(metrics):
    '''
    Summarize variation across support seeds, not confidence intervals.
    '''
    return {key: {"mean": float(np.mean([m[key] for m in metrics])),
                  "std": float(np.std([m[key] for m in metrics]))}
            for key in ("accuracy", "macro_f1", "top_3_accuracy")}
