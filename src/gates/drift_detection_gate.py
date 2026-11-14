import argparse
import logging
from pathlib import Path

import mlflow

from data.preprocessing import infer_schema, load_baseline_data, load_data
from drift.config.load import load_drift_config
from drift.detection import run_drift_checks
from utils.logging_utils import configure_mlflow, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drift Detection Gate")
    p.add_argument(
        "--gold-data-version",
        type=str,
        required=True,
        help="Mounted input path for gold star data",
    )
    p.add_argument(
        "--new-data-version",
        type=str,
        required=True,
        help="Mounted input path for new data to check",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/drift_config.yaml"),
        help="PSI threshold. Defaults to 0.25",
    )
    p.add_argument(
        "--drift-output-path",
        type=str,
        required=True,
        help="Path to write drift detection results",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> bool:
    setup_logging()
    configure_mlflow()

    drift_config = load_drift_config(args.config)

    mlflow.set_tag("gate", "drift-detection")

    baseline_stats = load_baseline_data(args.new_data_version)
    logger.info(f"Baseline data: {baseline_stats}")

    # Step 1: Load data asset ----------------
    train, _, _ = load_data(args.new_data_version)
    logger.info(f"New Data Shape: {train.shape}")

    logger.info("Inferring schema from baseline")
    num_cols, _ = infer_schema(train)

    drift_results, drifted_features, missing_columns = run_drift_checks(
        train, num_cols, drift_config, baseline_stats
    )

    # ---- Compute PSI and KS per feature
    mlflow.log_dict(drift_results, "drift_results/drift_results.json")

    max_psi = max((r["psi"] for r in drift_results.values()), default=0.0)
    n_drifted = len(drifted_features)
    stable = n_drifted == 0

    mlflow.log_metric("columns_missing", missing_columns)
    mlflow.log_metric("max_psi", max_psi)
    mlflow.log_metric("num_drifted_columns", n_drifted)
    mlflow.log_metric("stability", stable)

    if stable:
        logger.info(f"Drift gate PASSED - all features stable (max PSI={max_psi:.4f}")
    else:
        logger.error(
            f"Drift gate FAILED - {n_drifted} feature(s) drifted: {drifted_features}"
        )

    output_path = Path(args.drift_output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "drift.txt").write_text("stable" if stable else "drifted")
    logger.info(f"Wrote drift result to folder: {stable}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
