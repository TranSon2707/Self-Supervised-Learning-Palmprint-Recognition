"""
Benchmark pipeline for palmprint recognition.

Builds genuine and impostor pairs from the pre-computed embedding database,
scores them with cosine similarity, then delegates to `client.metrics` for
FAR / FRR / Valid Accuracy / EER computation.
"""

import re
import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from client.metrics import compute_metrics_at_threshold, compute_threshold_curve

_DEFAULT_EMBEDDINGS   = "output/embeddings/all_embeddings.npy"
_DEFAULT_IMAGE_NAMES  = "output/embeddings/image_names.npy"


# ── identity helpers ──────────────────────────────────────────────────────────

def extract_identity(image_name: str) -> str:
    """
    Derives a person-identity key from an image filename.

    Naming convention used in this project:
        'session1_00042'  →  '00042'
        'session2_00042'  →  '00042'   (same person, different session)

    Falls back to the full filename if no trailing digits are found.
    """
    m = re.search(r"(\d+)$", image_name)
    return m.group(1) if m else image_name


def _group_by_identity(image_names: List[str]) -> Dict[str, List[int]]:
    """Maps each identity key to the list of embedding-array indices."""
    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(image_names):
        key = extract_identity(name)
        groups.setdefault(key, []).append(idx)
    return groups


# ── pair builders ─────────────────────────────────────────────────────────────

def build_genuine_pairs(image_names: List[str]) -> List[Tuple[int, int]]:
    """
    Returns every (i, j) index pair where i and j share the same identity.
    Requires at least 2 images per identity.
    """
    groups = _group_by_identity(image_names)
    pairs: List[Tuple[int, int]] = []
    for indices in groups.values():
        if len(indices) >= 2:
            pairs.extend(combinations(indices, 2))
    return pairs


def build_impostor_pairs(
    image_names: List[str],
    n_pairs: int,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """
    Randomly samples `n_pairs` impostor (cross-identity) index pairs.
    Sampling is balanced with the number of genuine pairs by default.
    """
    rng = np.random.default_rng(seed)
    groups = _group_by_identity(image_names)
    ids = list(groups.keys())

    pairs: List[Tuple[int, int]] = []
    while len(pairs) < n_pairs:
        a, b = rng.choice(len(ids), size=2, replace=False)
        i = int(rng.choice(groups[ids[int(a)]]))
        j = int(rng.choice(groups[ids[int(b)]]))
        pairs.append((i, j))
    return pairs


# ── scoring ───────────────────────────────────────────────────────────────────

def score_pairs(
    embeddings: np.ndarray,
    pairs: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Vectorised batch cosine similarity for a list of (i, j) index pairs.

    Returns a 1-D array of similarity scores in [-1, 1].
    """
    if not pairs:
        return np.array([], dtype=float)

    ia = np.fromiter((p[0] for p in pairs), dtype=int, count=len(pairs))
    ib = np.fromiter((p[1] for p in pairs), dtype=int, count=len(pairs))

    a = embeddings[ia]
    b = embeddings[ib]
    dots  = np.sum(a * b, axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-10
    return dots / norms


# ── main entry point ──────────────────────────────────────────────────────────

def run_benchmark(
    embeddings_path: str           = _DEFAULT_EMBEDDINGS,
    image_names_path: str          = _DEFAULT_IMAGE_NAMES,
    threshold: float               = 0.8,
    max_impostors: Optional[int]   = None,
    seed: int                      = 42,
) -> dict:
    """
    End-to-end benchmark evaluation.

    Pipeline
    --------
    1. Load pre-computed embeddings and image names.
    2. Group images by identity; build genuine and impostor pairs.
    3. Compute cosine similarity for every pair (vectorised).
    4. Compute FAR, FRR, Valid Accuracy at *threshold*.
    5. Sweep thresholds across the full score range to find the EER.

    Parameters
    ----------
    embeddings_path  : path to all_embeddings.npy
    image_names_path : path to image_names.npy
    threshold        : operating decision threshold (default 0.8)
    max_impostors    : cap on impostor pairs; defaults to # genuine pairs
    seed             : RNG seed for reproducible impostor sampling

    Returns
    -------
    dict with keys
    ─────────────
    operating_threshold  – threshold used for point metrics
    FAR                  – False Acceptance Rate (%) at threshold
    FRR                  – False Rejection Rate (%) at threshold
    Valid_Accuracy       – Valid Accuracy (%) at threshold
    EER                  – Equal Error Rate (%)
    EER_threshold        – threshold at which EER occurs
    true_acceptances     – TP count (genuine pairs accepted)
    false_rejections     – FN count (genuine pairs rejected)
    false_acceptances    – FP count (impostor pairs accepted)
    true_rejections      – TN count (impostor pairs rejected)
    total_genuine        – total genuine pairs evaluated
    total_impostors      – total impostor pairs evaluated
    curve_thresholds     – threshold sweep array  (for plotting)
    curve_FAR            – FAR at each sweep threshold
    curve_FRR            – FRR at each sweep threshold
    genuine_scores       – raw scores for genuine pairs
    impostor_scores      – raw scores for impostor pairs
    """
    embeddings  = np.load(embeddings_path)
    image_names = np.load(image_names_path).tolist()

    genuine_pairs  = build_genuine_pairs(image_names)
    n_imp          = max_impostors if max_impostors is not None else len(genuine_pairs)
    impostor_pairs = build_impostor_pairs(image_names, n_imp, seed=seed)

    genuine_scores  = score_pairs(embeddings, genuine_pairs)
    impostor_scores = score_pairs(embeddings, impostor_pairs)

    pt  = compute_metrics_at_threshold(genuine_scores, impostor_scores, threshold)
    crv = compute_threshold_curve(genuine_scores, impostor_scores)

    return {
        # point metrics
        "operating_threshold": threshold,
        "FAR":                 pt["FAR"],
        "FRR":                 pt["FRR"],
        "Valid_Accuracy":      pt["Valid_Accuracy"],
        "true_acceptances":    pt["true_acceptances"],
        "false_rejections":    pt["false_rejections"],
        "false_acceptances":   pt["false_acceptances"],
        "true_rejections":     pt["true_rejections"],
        "total_genuine":       pt["total_genuine"],
        "total_impostors":     pt["total_impostors"],
        # EER (from full sweep)
        "EER":                 crv["EER"],
        "EER_threshold":       crv["EER_threshold"],
        # curve arrays for plotting
        "curve_thresholds":    crv["thresholds"],
        "curve_FAR":           crv["FAR"],
        "curve_FRR":           crv["FRR"],
        # raw scores for downstream use (e.g. dynamic threshold in UI)
        "genuine_scores":      genuine_scores,
        "impostor_scores":     impostor_scores,
    }
