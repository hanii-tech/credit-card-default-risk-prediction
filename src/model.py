"""Train and evaluate baseline models for credit-card default prediction.

Run from the project root with:

    python src/model.py
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_credit_card_data.csv"
TARGET_COLUMN = "default"
TEST_SIZE = 0.20
RANDOM_STATE = 42
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the cleaned dataset and separate features from the target."""
    data = pd.read_csv(path)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible, stratified 80/20 train-test split."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_models() -> dict[str, object]:
    """Create the baseline classifiers used in the comparison."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(
    model: object, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, object]:
    """Return classification metrics for a fitted model."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
    }


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, dict[str, object]]:
    """Fit every baseline model and evaluate it on the held-out test set."""
    results = {}

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)

    return results


def print_results(results: dict[str, dict[str, object]]) -> None:
    """Print the requested metrics in a readable format."""
    print("\nModel evaluation results:")
    print("=" * 80)

    for name, metrics in results.items():
        print(f"\n{name}")
        print("-" * len(name))
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1-score:    {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
        print("Confusion matrix [ [true negative, false positive],")
        print("                  [false negative, true positive] ]:")
        print(metrics["confusion_matrix"])


def save_performance_plots(
    results: dict[str, dict[str, object]], output_dir: Path = FIGURE_DIR
) -> None:
    """Save confusion matrices and a combined ROC curve comparison."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        1, len(results), figsize=(15, 4.5), constrained_layout=True
    )
    for axis, (name, metrics) in zip(axes, results.items()):
        matrix = metrics["confusion_matrix"]
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1], ["No default", "Default"])
        axis.set_yticks([0, 1], ["No default", "Default"])

        threshold = matrix.max() / 2
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                    fontsize=12,
                )

    figure.colorbar(image, ax=axes, shrink=0.8, label="Number of customers")
    figure.suptitle("Confusion matrices on the test set")
    figure.savefig(output_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for name, metrics in results.items():
        axis.plot(
            metrics["false_positive_rate"],
            metrics["true_positive_rate"],
            label=f"{name} (AUC = {metrics['roc_auc']:.3f})",
        )

    axis.plot([0, 1], [0, 1], "k--", label="Random baseline")
    axis.set_title("ROC curves")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"\nPerformance plots saved to: {output_dir}")


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Dataset: {len(X)} customers, {X.shape[1]} features")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")

    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    print_results(results)
    save_performance_plots(results)


if __name__ == "__main__":
    main()