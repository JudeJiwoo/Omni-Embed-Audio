"""
Evaluation Metrics for Audio Retrieval.

Provides common metrics for retrieval evaluation:
- Recall@K
- Mean Reciprocal Rank (MRR)
- Discounted Cumulative Gain (DCG)
- R-sum (sum of recalls)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def validate_embeddings(embeddings: np.ndarray, source: str) -> None:
    """Reject non-finite external embeddings before normalization or ranking.

    Validate embeddings at the input boundary, not similarity scores: ranking
    deliberately uses -inf to exclude candidates.
    """
    if not np.isfinite(embeddings).all():
        nan_count = np.count_nonzero(np.isnan(embeddings))
        posinf_count = np.count_nonzero(np.isposinf(embeddings))
        neginf_count = np.count_nonzero(np.isneginf(embeddings))
        raise ValueError(
            f"Array 'embeddings' in '{source}' contains non-finite values "
            f"(NaN={nan_count}, +inf={posinf_count}, -inf={neginf_count}). "
            "Repair or regenerate the embedding file before evaluation."
        )


def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """
    L2 normalize array along specified axis.

    Args:
        x: Input array
        axis: Axis to normalize along
        eps: Small constant to avoid division by zero

    Returns:
        L2-normalized array
    """
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)


def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix.

    Assumes inputs are L2-normalized.

    Args:
        A: First embedding matrix [N, D]
        B: Second embedding matrix [M, D]

    Returns:
        Similarity matrix [N, M]
    """
    return A @ B.T


def ranks_from_scores(
    scores_row: np.ndarray,
    gt_index: int,
    ignore_index: Optional[int] = None,
) -> int:
    """
    Compute rank of ground truth item from similarity scores.

    Args:
        scores_row: Similarity scores for all items
        gt_index: Index of ground truth item
        ignore_index: Optional index to ignore in ranking

    Returns:
        Rank of ground truth item (1-indexed)
    """
    if ignore_index is not None and 0 <= ignore_index < scores_row.shape[0] and ignore_index != gt_index:
        tmp = scores_row.copy()
        tmp[ignore_index] = -np.inf
        gt_score = tmp[gt_index]
        return int(1 + np.sum(tmp > gt_score))
    gt_score = scores_row[gt_index]
    return int(1 + np.sum(scores_row > gt_score))


def compute_recall_at_k(ranks: np.ndarray, k: int) -> float:
    """
    Compute Recall@K metric.

    Args:
        ranks: Array of ranks for each query
        k: K value for recall computation

    Returns:
        Recall@K as percentage (0-100)
    """
    return float(np.mean(ranks <= k) * 100.0)


def compute_mrr(ranks: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        ranks: Array of ranks for each query

    Returns:
        MRR value
    """
    return float(np.mean(1.0 / ranks))


def compute_dcg(ranks: np.ndarray) -> float:
    """
    Compute Discounted Cumulative Gain.

    Args:
        ranks: Array of ranks for each query

    Returns:
        DCG value
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    gains = 1.0 / np.log2(ranks + 1.0)
    return float(np.mean(gains))


def compute_rsum(recalls: Dict[str, float], keys: Sequence[str] = ("R@1", "R@5", "R@10")) -> float:
    """
    Compute sum of recall metrics.

    Args:
        recalls: Dictionary of recall metrics
        keys: Keys to sum

    Returns:
        Sum of specified recall values
    """
    return float(sum(recalls.get(k, 0.0) for k in keys))


def compute_all_metrics(ranks: np.ndarray) -> Dict[str, float]:
    """
    Compute all standard retrieval metrics.

    Args:
        ranks: Array of ranks for each query

    Returns:
        Dictionary with R@1, R@5, R@10, MRR, DCG
    """
    return {
        "R@1": compute_recall_at_k(ranks, 1),
        "R@5": compute_recall_at_k(ranks, 5),
        "R@10": compute_recall_at_k(ranks, 10),
        "MRR": compute_mrr(ranks),
        "DCG": compute_dcg(ranks),
    }


def format_metrics(metrics: Dict[str, float], precision: int = 2) -> str:
    """
    Format metrics dictionary as readable string.

    Args:
        metrics: Dictionary of metric values
        precision: Decimal precision for formatting

    Returns:
        Formatted string
    """
    parts = []
    for key in ["R@1", "R@5", "R@10", "MRR", "DCG"]:
        if key in metrics:
            if key.startswith("R@"):
                parts.append(f"{key}={metrics[key]:.{precision}f}%")
            else:
                parts.append(f"{key}={metrics[key]:.{precision+2}f}")
    return " | ".join(parts)


class MetricsAggregator:
    """
    Aggregator for computing metrics across multiple evaluation runs.

    Example:
        >>> agg = MetricsAggregator()
        >>> agg.add("text2audio", ranks_t2a)
        >>> agg.add("audio2text", ranks_a2t)
        >>> results = agg.compute()
    """

    def __init__(self):
        self._ranks: Dict[str, List[np.ndarray]] = {}

    def add(self, name: str, ranks: np.ndarray) -> None:
        """Add ranks for a named task."""
        if name not in self._ranks:
            self._ranks[name] = []
        self._ranks[name].append(ranks)

    def compute(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics for all added tasks."""
        results = {}
        for name, rank_list in self._ranks.items():
            all_ranks = np.concatenate(rank_list)
            results[name] = compute_all_metrics(all_ranks)
        return results

    def reset(self) -> None:
        """Reset aggregator."""
        self._ranks.clear()
