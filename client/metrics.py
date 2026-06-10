import numpy as np
from typing import Tuple


# ── individual metric formulas ────────────────────────────────────────────────

def calculate_far(false_acceptances: int, total_impostors: int) -> float:
    """
    False Acceptance Rate — percentage of impostor attempts incorrectly accepted.
    FAR = (false acceptances / total impostor instances) * 100
    """
    if total_impostors == 0:
        return 0.0
    return (false_acceptances / total_impostors) * 100.0


def calculate_frr(false_rejections: int, total_genuine: int) -> float:
    """
    False Rejection Rate — percentage of genuine attempts incorrectly rejected.
    FRR = (false rejections / total genuine attempts) * 100
    """
    if total_genuine == 0:
        return 0.0
    return (false_rejections / total_genuine) * 100.0


def calculate_valid_accuracy(true_acceptances: int, total_genuine: int) -> float:
    """
    Valid Accuracy — percentage of genuine attempts correctly accepted.
    Valid Accuracy = (true acceptances / total genuine instances) * 100
    """
    if total_genuine == 0:
        return 0.0
    return (true_acceptances / total_genuine) * 100.0


def calculate_eer(
    far_values: np.ndarray,
    frr_values: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float, int]:
    """
    Equal Error Rate — threshold at which FAR and FRR are closest.
    EER = (FAR + FRR) / 2  at the crossover point.

    Returns
    -------
    eer          : EER value as a percentage
    eer_threshold: decision threshold at which EER occurs
    eer_idx      : index into the thresholds array
    """
    diff = np.abs(np.asarray(far_values) - np.asarray(frr_values))
    idx = int(np.argmin(diff))
    eer = (far_values[idx] + frr_values[idx]) / 2.0
    return float(eer), float(thresholds[idx]), idx


# ── aggregated metric helpers ─────────────────────────────────────────────────

def compute_metrics_at_threshold(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
) -> dict:
    """
    Computes FAR, FRR, Valid Accuracy, and raw counts at a single threshold.

    Decision rule: score >= threshold → accepted, score < threshold → rejected.

    Parameters
    ----------
    genuine_scores  : similarity scores for genuine pairs  (same identity)
    impostor_scores : similarity scores for impostor pairs (different identity)
    threshold       : decision threshold in [0, 1]

    Returns
    -------
    dict with keys:
        threshold, FAR, FRR, Valid_Accuracy,
        true_acceptances, false_rejections, false_acceptances, true_rejections,
        total_genuine, total_impostors
    """
    g = np.asarray(genuine_scores)
    n = np.asarray(impostor_scores)

    total_genuine   = len(g)
    total_impostors = len(n)

    # Genuine pair outcomes
    true_acceptances  = int(np.sum(g >= threshold))   # correctly accepted
    false_rejections  = int(np.sum(g < threshold))    # incorrectly rejected

    # Impostor pair outcomes
    false_acceptances = int(np.sum(n >= threshold))   # incorrectly accepted
    true_rejections   = int(np.sum(n < threshold))    # correctly rejected

    return {
        "threshold":         threshold,
        "FAR":               calculate_far(false_acceptances, total_impostors),
        "FRR":               calculate_frr(false_rejections, total_genuine),
        "Valid_Accuracy":    calculate_valid_accuracy(true_acceptances, total_genuine),
        "true_acceptances":  true_acceptances,
        "false_rejections":  false_rejections,
        "false_acceptances": false_acceptances,
        "true_rejections":   true_rejections,
        "total_genuine":     total_genuine,
        "total_impostors":   total_impostors,
    }


def compute_threshold_curve(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    num_steps: int = 300,
) -> dict:
    """
    Sweeps decision thresholds across the full score range and records
    FAR and FRR at each step. Locates the EER.

    Returns
    -------
    dict with keys:
        thresholds (array), FAR (array), FRR (array),
        EER (float), EER_threshold (float)
    """
    g = np.asarray(genuine_scores)
    n = np.asarray(impostor_scores)

    lo = float(min(g.min(), n.min()))
    hi = float(max(g.max(), n.max()))
    thresholds = np.linspace(lo, hi, num_steps)

    ng = len(g)
    ni = len(n)
    far_curve = np.empty(num_steps)
    frr_curve = np.empty(num_steps)

    for i, t in enumerate(thresholds):
        far_curve[i] = calculate_far(int(np.sum(n >= t)), ni)
        frr_curve[i] = calculate_frr(int(np.sum(g < t)), ng)

    eer, eer_thr, _ = calculate_eer(far_curve, frr_curve, thresholds)

    return {
        "thresholds":    thresholds,
        "FAR":           far_curve,
        "FRR":           frr_curve,
        "EER":           eer,
        "EER_threshold": eer_thr,
    }
