"""CLI entrypoint for training, predicting, and evaluating the fraud model bundle.

Commands:
- train: train a RandomForest model and export a pickled `ModelBundle`.
- predict: load a saved bundle and score a CSV of features.
- eval: load a saved bundle and evaluate it on a labeled CSV.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.training import train_random_forest, ModelBundle
from src.inference import predict_dataframe, evaluate_on_labeled_csv


def save_bundle_pickle(bundle: ModelBundle, path: Path) -> None:
    """Serialize and save a `ModelBundle` to disk using pickle.

    Args:
        bundle: The model bundle to serialize.
        path: Output path for the pickle file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bundle, f)


def load_bundle_pickle(path: Path) -> ModelBundle:
    """Load a pickled `ModelBundle` from disk.

    Args:
        path: Path to the pickle file.

    Returns:
        The deserialized `ModelBundle`.

    Raises:
        TypeError: If the pickle does not contain a `ModelBundle`.
    """
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, ModelBundle):
        raise TypeError(f"Pickle does not contain ModelBundle, got: {type(obj)}")
    return obj


def cmd_train(args: argparse.Namespace) -> None:
    """Train a model bundle from a labeled CSV and save it to disk.

    Args:
        args: Parsed CLI arguments containing dataset path, output path, and split params.
    """
    bundle, metrics = train_random_forest(
        train_path=Path(args.data),
        test_size=args.test_size,
        seed=args.seed,
    )

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    out_path = Path(args.out)
    save_bundle_pickle(bundle, out_path)
    print(f"\nSaved model bundle to: {out_path.resolve()}")


def cmd_predict(args: argparse.Namespace) -> None:
    """Load a saved bundle and score a CSV file containing feature columns.

    Args:
        args: Parsed CLI arguments containing model path, input CSV path, and output path.
    """
    bundle = load_bundle_pickle(Path(args.model))

    df = pd.read_csv(args.data)
    scored = predict_dataframe(bundle, df)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out_path, index=False)
        print(f"Saved predictions to: {out_path.resolve()}")
    else:
        print(scored.head(20).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        An `ArgumentParser` configured with `train`, `predict`, and `eval` subcommands.
    """
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train model and export to pickle")
    p_train.add_argument(
        "--data", required=True, help="Path to Train CSV (must include target column)")
    p_train.add_argument("--out", default="models/fraud_rf_bundle.pkl", help="Output pickle path")
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=1337)
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Load pickle model and score new data")
    p_pred.add_argument(
        "--model", default="models/fraud_rf_bundle.pkl", help="Path to saved pickle bundle")
    p_pred.add_argument("--data", required=True, help="Path to CSV with features only")
    p_pred.add_argument(
        "--out", default="data/scored.csv", help="Where to save scored CSV (empty to print)")
    p_pred.set_defaults(func=cmd_predict)

    p_eval = sub.add_parser(
        "eval", help="Evaluate a saved pickle model on labeled test CSV (with target column)")
    p_eval.add_argument(
        "--model", default="models/fraud_rf_bundle.pkl", help="Path to saved pickle bundle")
    p_eval.add_argument(
        "--data", required=True, help="Path to TEST CSV (must include target column)")
    p_eval.set_defaults(func=cmd_eval)

    return p

def cmd_eval(args: argparse.Namespace) -> None:
    """Load a saved bundle and evaluate it on a labeled CSV dataset.

    Args:
        args: Parsed CLI arguments containing model path and labeled CSV path.
    """
    bundle = load_bundle_pickle(Path(args.model))
    df = pd.read_csv(args.data)

    metrics = evaluate_on_labeled_csv(bundle, df)

    print("\n=== Evaluation ===")
    print(f"AUC: {metrics['auc']}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"N: {metrics['n']}")
    print(f"Positive rate (true): {metrics['positive_rate_true']}")
    print(f"Positive rate (pred, thr=0.5): {metrics['positive_rate_pred']}")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])

    print("\nClassification report:")
    print(metrics["classification_report"])

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
