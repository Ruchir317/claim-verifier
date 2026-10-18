"""Score the verifier's True/False verdicts against PolitiFact's own ratings.

    python evaluate_accuracy.py

PolitiFact uses a six-point scale; it is collapsed to True/False so it can be compared with the model:
true, mostly-true and half-true count as True; barely-true, false and pants-fire count as False.
Use --true-labels to change that mapping.
"""
import argparse
import csv
import json
import os

DEFAULT_TRUE_LABELS = ("true", "mostly-true", "half-true")


def load_json_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_truth_lookup(politifact_rows, true_labels=DEFAULT_TRUE_LABELS):
    labels = {label.lower().strip() for label in true_labels}
    return {row["statement"]: row["verdict"].lower().strip() in labels for row in politifact_rows}


def compare(model_outputs, truth_lookup):
    """Pair each True/False model verdict with PolitiFact's. Returns (rows, unmatched_statements)."""
    rows, unmatched = [], []
    for entry in model_outputs:
        verdict = entry.get("verdict")
        if verdict not in ("True", "False"):
            continue  # malformed or missing verdict
        statement = entry["fact"]
        if statement not in truth_lookup:
            unmatched.append(statement)
            continue
        rows.append(
            {
                "statement": statement,
                "actual_verdict": truth_lookup[statement],
                "model_verdict": verdict == "True",
            }
        )
    return rows, unmatched


def accuracy(rows):
    return sum(r["actual_verdict"] == r["model_verdict"] for r in rows) / len(rows) if rows else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate verdicts against PolitiFact ratings.")
    parser.add_argument("--parsed", default="output/parsed_output.json")
    parser.add_argument("--data", default="data/politifact_factcheck_data.json")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--true-labels", nargs="+", default=list(DEFAULT_TRUE_LABELS))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")  # write the image, never open a window
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    with open(args.parsed, "r", encoding="utf-8") as f:
        model_outputs = json.load(f)
    truth = build_truth_lookup(load_json_lines(args.data), args.true_labels)
    rows, unmatched = compare(model_outputs, truth)
    if not rows:
        raise SystemExit("No comparable verdicts found. Run batch_fact_verifier.py first.")

    csv_path = os.path.join(args.output_dir, "verdict_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["statement", "actual_verdict", "model_verdict"])
        writer.writeheader()
        writer.writerows(rows)

    y_true = [r["actual_verdict"] for r in rows]
    y_pred = [r["model_verdict"] for r in rows]
    labels = ["False", "True"]
    print(f"Compared {len(rows)} claims ({len(unmatched)} not found in the dataset).")
    print(f"Accuracy: {accuracy(rows):.4f}")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("PolitiFact")
    plt.tight_layout()
    image_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(image_path)
    print(f"Wrote {csv_path} and {image_path}")


if __name__ == "__main__":
    main()
