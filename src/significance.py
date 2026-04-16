"""Statistical significance testing module.

Provides bootstrap confidence intervals and McNemar's test for
comparing classifier predictions. These are required for any academic
claim that model A is better than model B.

Improvements:
  - Added Bonferroni correction for multiple pairwise McNemar tests.
  - Corrected p-value interpretation labels to include corrected threshold.
  - bootstrap_confidence_interval now also reports median and IQR.
"""

from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.metrics import classification_metrics


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str],
    metric_name: str = "macro_f1",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> Dict[str, object]:
    """Estimate a confidence interval for a scalar metric via bootstrap resampling.

    Args:
        y_true: Integer class labels (ground truth).
        y_prob: Probability array (n_samples, n_classes).
        labels: Class label names.
        metric_name: Key from classification_metrics() to extract.
        n_bootstrap: Number of bootstrap resamples (1000 is standard).
        confidence: Confidence level (default 0.95 → 95% CI).
        random_state: RNG seed for reproducibility.

    Returns:
        Dictionary with mean, lower_ci, upper_ci, std, median, n_bootstrap.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        p_boot = y_prob[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        try:
            m = classification_metrics(y_boot, p_boot, labels)
            val = m[metric_name]
            if isinstance(val, float) and not np.isnan(val):
                scores.append(val)
        except Exception:
            continue

    if not scores:
        return {
            "metric": metric_name,
            "mean": float("nan"),
            "median": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "std": float("nan"),
            "iqr_lower": float("nan"),
            "iqr_upper": float("nan"),
            "n_bootstrap": n_bootstrap,
            "confidence": confidence,
            "n_valid_samples": 0,
        }

    alpha = 1.0 - confidence
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1.0 - alpha / 2)))
    return {
        "metric": metric_name,
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "lower_ci": lower,
        "upper_ci": upper,
        "std": float(np.std(scores)),
        "iqr_lower": float(np.percentile(scores, 25)),
        "iqr_upper": float(np.percentile(scores, 75)),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "n_valid_samples": len(scores),
    }


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    bonferroni_n: int = 1,
) -> Dict[str, object]:
    """McNemar's test comparing two classifiers on the same test set.

    A statistically significant result (p < 0.05) means one model is
    genuinely better than the other.

    Args:
        y_true: Ground-truth class labels.
        preds_a: Hard predictions from model A.
        preds_b: Hard predictions from model B.
        model_a_name: Name for reporting.
        model_b_name: Name for reporting.
        bonferroni_n: Number of comparisons for Bonferroni correction.
                      corrected_alpha = 0.05 / bonferroni_n.

    Returns:
        Dict with statistic, p_value, corrected_p_value, contingency_table, interpretation.
    """
    a_correct = (preds_a == y_true)
    b_correct = (preds_b == y_true)

    n00 = int(np.sum(~a_correct & ~b_correct))
    n01 = int(np.sum(~a_correct & b_correct))
    n10 = int(np.sum(a_correct & ~b_correct))
    n11 = int(np.sum(a_correct & b_correct))

    b = n01
    c = n10
    if (b + c) == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        # With continuity correction (Yates)
        statistic = float(((abs(b - c) - 1) ** 2) / (b + c))
        from scipy import stats
        p_value = float(1.0 - stats.chi2.cdf(statistic, df=1))

    # Bonferroni correction
    corrected_alpha = 0.05 / max(bonferroni_n, 1)

    def _sig_label(p: float, alpha: float) -> str:
        if p < 0.001:
            return f"highly_significant (p < 0.001, alpha={alpha:.4f})"
        if p < 0.01:
            return f"significant (p < 0.01, alpha={alpha:.4f})"
        if p < alpha:
            return f"significant (p < {alpha:.4f})"
        return f"not_significant (p >= {alpha:.4f})"

    raw_sig = _sig_label(p_value, 0.05)
    corrected_sig = _sig_label(p_value, corrected_alpha)
    winner = model_a_name if n10 > n01 else (model_b_name if n01 > n10 else "tie")

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "contingency_table": {
            "both_correct": n11,
            "a_correct_b_wrong": n10,
            "a_wrong_b_correct": n01,
            "both_wrong": n00,
        },
        "mcnemar_statistic": statistic,
        "p_value": p_value,
        "significance_raw": raw_sig,
        "bonferroni_n": bonferroni_n,
        "corrected_alpha": corrected_alpha,
        "significance_bonferroni_corrected": corrected_sig,
        "significant_after_bonferroni": bool(p_value < corrected_alpha),
        "apparent_winner": winner,
        "interpretation": (
            f"p={p_value:.4f} (Bonferroni-corrected alpha={corrected_alpha:.4f}): "
            f"the difference between {model_a_name} and {model_b_name} "
            f"is {corrected_sig}."
        ),
    }


def run_significance_suite(
    y_true: np.ndarray,
    model_results: List[object],
    labels: List[str],
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, object]:
    """Run full significance suite: bootstrap CIs + all pairwise McNemar tests with Bonferroni.

    Args:
        y_true: Ground-truth labels for the test set.
        model_results: List of ModelResult objects — must be evaluated on the SAME test set.
        labels: Class label names.
        n_bootstrap: Bootstrap resamples.
        random_state: Seed.

    Returns:
        Dict with bootstrap_cis and mcnemar_pairwise sections.
    """
    output: Dict[str, object] = {}

    # Bootstrap CIs per model
    ci_rows = []
    for result in model_results:
        if result.estimator is None:
            continue
        proba = result.validation_probabilities
        if proba is None or len(proba) != len(y_true):
            continue
        for metric_name in ["macro_f1", "macro_f1_present_classes", "balanced_accuracy", "log_loss"]:
            ci = bootstrap_confidence_interval(
                y_true, proba, labels,
                metric_name=metric_name,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            ci_rows.append({"model": result.name, **ci})
    output["bootstrap_cis"] = ci_rows

    # Pairwise McNemar tests with Bonferroni correction
    valid_results = [
        r for r in model_results
        if r.estimator is not None
        and r.validation_probabilities is not None
        and len(r.validation_probabilities) == len(y_true)
    ]
    n_pairs = len(list(combinations(range(len(valid_results)), 2)))
    mcnemar_rows = []
    for i in range(len(valid_results)):
        for j in range(i + 1, len(valid_results)):
            r_a = valid_results[i]
            r_b = valid_results[j]
            preds_a = np.argmax(r_a.validation_probabilities, axis=1)
            preds_b = np.argmax(r_b.validation_probabilities, axis=1)
            result = mcnemar_test(
                y_true, preds_a, preds_b, r_a.name, r_b.name,
                bonferroni_n=n_pairs,   # Bonferroni correction applied here
            )
            mcnemar_rows.append(result)
    output["mcnemar_pairwise"] = mcnemar_rows
    output["bonferroni_note"] = (
        f"Bonferroni correction applied across {n_pairs} pairwise comparisons. "
        f"Corrected alpha = {0.05 / max(n_pairs, 1):.4f}. "
        "Use 'significant_after_bonferroni' field for the corrected conclusion."
    )

    return output
