from drift.config.drift_config import DriftConfig
import numpy as np
from scipy import stats
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def psi_for_column(
    baseline: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-6
) -> float:
    """
    Compute Population Stability Index between baseline and current distributions.
    PSI < 0.10  → No significant change
    PSI 0.10–0.20 → Moderate change (warning)
    PSI > 0.20  → Significant change (drift)
    """
    global_min = min(baseline.min(), current.min())
    global_max = max(baseline.max(), current.max())
    bins = np.linspace(global_min, global_max, n_bins + 1)

    baseline_counts, _ = np.histogram(baseline, bins=bins)
    current_counts, _ = np.histogram(current, bins=bins)

    baseline_pct = (baseline_counts / len(baseline)) + eps
    current_pct = (current_counts / len(current)) + eps

    psi = float(
        np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    )
    return psi


def ks_test(baseline: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Return (statistic, p_value) from the KS two-sample test."""
    result = stats.ks_2samp(baseline, current)
    return float(result.statistic), float(result.pvalue)


def run_drift_checks(
    train: pd.DataFrame,
    num_cols: list[str],
    drift_config: DriftConfig,
    baseline_stats: dict[str, float],
) -> tuple[dict[str, dict], list[str], int]:
    """
    Run drift checks for all numerical features in training data, compared to baseline information in all data
    """

    drift_results: dict[str, dict] = {}
    drifted_features: list[str] = []
    missing_columns = 0

    for col in num_cols:
        if col not in train.columns:
            logger.warning(f"Column {col} missing from current data - skipping")
            missing_columns += 1
            continue

        current_vals = train[col].dropna().values
        bstats = baseline_stats[col]
        rng = np.random.default_rng(drift_config.drift.rng)
        baseline_vals = rng.normal(
            loc=bstats["mean"],
            scale=max(bstats["std"], drift_config.drift.scale),
            size=bstats["n"],
        )

        psi = psi_for_column(
            baseline_vals, current_vals, drift_config.psi.n_bins, drift_config.psi.eps
        )
        ks_stat, ks_pval = ks_test(baseline_vals, current_vals)

        drift_results[col] = {
            "psi": psi,
            "ks_stat": ks_stat,
            "ks_pval": ks_pval,
            "drifted": psi > drift_config.drift.threshold,
        }

        if psi > drift_config.drift.threshold:
            drifted_features.append(col)
            logger.warning(
                f"DRIFT detected in {col}: PSI={psi:.4f} > threshold={drift_config.drist.threshold:.4f}"
            )
        else:
            logger.info(f"Stable: {col} PSI={psi:.4f}")

    return drift_results, drifted_features, missing_columns
