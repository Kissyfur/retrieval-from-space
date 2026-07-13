from __future__ import annotations

import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def labels_from_probabilities(probabilities, class_index: int | None = None, threshold: float | None = None):
    probabilities = np.asarray(probabilities)
    if probabilities.ndim <= 1 or probabilities.shape[1] <= 1:
        return probabilities.reshape(-1)
    if class_index is None or threshold is None:
        return np.argmax(probabilities, axis=1)

    class_index = int(class_index)
    if class_index < 0 or class_index >= probabilities.shape[1]:
        raise ValueError(
            f"class_index must be between 0 and {probabilities.shape[1] - 1}; got {class_index}."
        )
    fallback = probabilities.copy()
    fallback[:, class_index] = -np.inf
    labels = np.argmax(fallback, axis=1)
    labels[probabilities[:, class_index] >= float(threshold)] = class_index
    return labels


def classification_metrics(y_true, y_pred, labels=None) -> dict[str, object]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def threshold_curve_metrics(y_true, probabilities, class_index: int, thresholds, labels=None):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(np.argmax(probabilities, axis=1).tolist()))
    if class_index not in labels:
        labels = [*labels, class_index]
    class_position = list(labels).index(class_index)

    rows = []
    for threshold in thresholds:
        y_pred = labels_from_probabilities(probabilities, class_index, float(threshold))
        per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision_macro": float(
                    precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
                ),
                "recall_macro": float(
                    recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
                ),
                "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
                "precision_class": float(per_class_precision[class_position]),
                "recall_class": float(per_class_recall[class_position]),
                "f1_class": float(per_class_f1[class_position]),
                "predicted_class_count": int(np.sum(y_pred == class_index)),
            }
        )
    return rows


def save_confusion_matrix_plot(
    y_true,
    y_pred,
    path,
    labels=None,
    normalize: str | None = None,
    title: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title or "Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)), labels=[str(label) for label in labels])
    ax.set_yticks(np.arange(len(labels)), labels=[str(label) for label in labels])

    threshold = float(np.nanmax(matrix)) / 2 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            color = "white" if value > threshold else "black"
            ax.text(col, row, text, ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(path, format="jpg", bbox_inches="tight")
    plt.close(fig)


def save_threshold_curve_plot(rows, path, class_index: int) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    splits = sorted({row.get("split", "test") for row in rows})
    for split in splits:
        split_rows = sorted(
            [row for row in rows if row.get("split", "test") == split],
            key=lambda row: row["threshold"],
        )
        if not split_rows:
            continue
        thresholds = [row["threshold"] for row in split_rows]
        ax.plot(thresholds, [row["f1_macro"] for row in split_rows], marker="o", label=f"{split} macro F1")
        ax.plot(
            thresholds,
            [row["f1_class"] for row in split_rows],
            marker="s",
            linestyle="--",
            label=f"{split} class {class_index} F1",
        )
    ax.set_xlabel(f"Class {class_index} probability threshold")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, format="jpg", bbox_inches="tight")
    plt.close(fig)
