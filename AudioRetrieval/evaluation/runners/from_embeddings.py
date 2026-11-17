"""
Precomputed Embedding Evaluation Runner.

Evaluates retrieval using precomputed embeddings (NPZ files).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from AudioRetrieval.evaluation.metrics import (
    validate_embeddings,
    l2norm,
    cosine_sim,
    ranks_from_scores,
    compute_all_metrics,
    format_metrics,
)


class PrecomputedEmbeddingRunner:
    """
    Evaluation runner using precomputed embeddings.

    Loads embeddings from NPZ files and evaluates retrieval.

    Args:
        audio_embeddings_path: Path to audio embeddings NPZ
        caption_embeddings_path: Optional path to caption embeddings NPZ
        uiq_embeddings_dir: Optional directory containing UIQ embeddings

    Example:
        >>> runner = PrecomputedEmbeddingRunner(
        ...     audio_embeddings_path="embeddings/audio.npz",
        ...     caption_embeddings_path="embeddings/captions.npz",
        ... )
        >>> results = runner.run_baseline()
    """

    def __init__(
        self,
        audio_embeddings_path: Path,
        caption_embeddings_path: Optional[Path] = None,
        uiq_embeddings_dir: Optional[Path] = None,
    ):
        self.audio_embeddings_path = Path(audio_embeddings_path)
        self.caption_embeddings_path = Path(caption_embeddings_path) if caption_embeddings_path else None
        self.uiq_embeddings_dir = Path(uiq_embeddings_dir) if uiq_embeddings_dir else None

        self._audio_data = None
        self._caption_data = None

    @property
    def audio_data(self) -> Dict[str, np.ndarray]:
        """Lazy-load audio embeddings."""
        if self._audio_data is None:
            data = np.load(self.audio_embeddings_path, allow_pickle=True)
            validate_embeddings(data["embeddings"], str(self.audio_embeddings_path))
            self._audio_data = {
                "embeddings": data["embeddings"],
                "clip_ids": list(data.get("clip_ids", [])),
                "filenames": list(data.get("filenames", [])),
            }
        return self._audio_data

    @property
    def caption_data(self) -> Optional[Dict[str, np.ndarray]]:
        """Lazy-load caption embeddings."""
        if self._caption_data is None and self.caption_embeddings_path:
            data = np.load(self.caption_embeddings_path, allow_pickle=True)
            validate_embeddings(data["embeddings"], str(self.caption_embeddings_path))
            self._caption_data = {
                "embeddings": data["embeddings"],
                "texts": list(data.get("texts", [])),
                "clip_ids": list(data.get("clip_ids", [])),
            }
        return self._caption_data

    def load_uiq_embeddings(self, category: str) -> Optional[Dict[str, np.ndarray]]:
        """Load UIQ embeddings for a specific category."""
        if not self.uiq_embeddings_dir:
            return None

        # Try different filename formats
        category_safe = category.lower().replace("/", "_").replace("-", "_")
        possible_names = [
            f"uiq_{category_safe}_embeddings.npz",
            f"uiq_{category_safe}.npz",
            f"{category_safe}_embeddings.npz",
        ]

        for name in possible_names:
            path = self.uiq_embeddings_dir / name
            if path.exists():
                data = np.load(path, allow_pickle=True)
                validate_embeddings(data["embeddings"], str(path))
                return {
                    "embeddings": data["embeddings"],
                    "texts": list(data.get("texts", [])),
                    "clip_ids": list(data.get("clip_ids", [])),
                    "bucket": str(data.get("bucket", category)),
                }

        return None

    def run_baseline(self) -> Dict[str, Dict[str, float]]:
        """
        Run baseline evaluation using precomputed embeddings.

        Returns:
            Dictionary with T2A and A2T metrics
        """
        if not self.caption_data:
            raise ValueError("Caption embeddings required for baseline evaluation")

        print("=" * 60)
        print("Baseline Evaluation (Precomputed Embeddings)")
        print("=" * 60)

        audio_embeds = l2norm(self.audio_data["embeddings"])
        text_embeds = l2norm(self.caption_data["embeddings"])

        audio_clip_ids = self.audio_data["clip_ids"]
        text_clip_ids = self.caption_data["clip_ids"]

        print(f"Audio embeddings: {audio_embeds.shape}")
        print(f"Text embeddings: {text_embeds.shape}")

        # Build audio index mapping
        audio_id_to_idx = {cid: idx for idx, cid in enumerate(audio_clip_ids)}

        # Text-to-Audio evaluation
        sim_matrix_t2a = cosine_sim(text_embeds, audio_embeds)
        t2a_ranks = []
        for i, clip_id in enumerate(text_clip_ids):
            if clip_id in audio_id_to_idx:
                gt_idx = audio_id_to_idx[clip_id]
                rank = ranks_from_scores(sim_matrix_t2a[i], gt_idx)
                t2a_ranks.append(rank)

        t2a_metrics = compute_all_metrics(np.array(t2a_ranks))

        # Audio-to-Text evaluation
        sim_matrix_a2t = cosine_sim(audio_embeds, text_embeds)
        a2t_ranks = []

        for audio_idx, audio_clip_id in enumerate(audio_clip_ids):
            # Find all text indices for this audio
            gt_text_indices = [i for i, cid in enumerate(text_clip_ids) if cid == audio_clip_id]
            if not gt_text_indices:
                continue

            # Get best rank
            scores = sim_matrix_a2t[audio_idx]
            best_rank = min(ranks_from_scores(scores, gt_idx) for gt_idx in gt_text_indices)
            a2t_ranks.append(best_rank)

        a2t_metrics = compute_all_metrics(np.array(a2t_ranks))

        print()
        print("Results:")
        print(f"  T2A: {format_metrics(t2a_metrics)}")
        print(f"  A2T: {format_metrics(a2t_metrics)}")

        return {
            "text_to_audio": t2a_metrics,
            "audio_to_text": a2t_metrics,
        }

    def run_uiq(
        self,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run UIQ evaluation using precomputed embeddings.

        Args:
            categories: List of UIQ categories to evaluate

        Returns:
            Dictionary with metrics for each category
        """
        if not self.uiq_embeddings_dir:
            raise ValueError("UIQ embeddings directory required for UIQ evaluation")

        if categories is None:
            categories = [
                "Pragmatic",
                "Subjective",
                "Contextual",
                "Negative/Contrastive",
                "Cross-domain",
            ]

        print("=" * 60)
        print("UIQ Evaluation (Precomputed Embeddings)")
        print("=" * 60)

        audio_embeds = l2norm(self.audio_data["embeddings"])
        audio_clip_ids = self.audio_data["clip_ids"]
        audio_id_to_idx = {cid: idx for idx, cid in enumerate(audio_clip_ids)}

        print(f"Audio embeddings: {audio_embeds.shape}")

        results = {}
        for category in categories:
            uiq_data = self.load_uiq_embeddings(category)
            if not uiq_data:
                print(f"Warning: UIQ category '{category}' not found")
                continue

            uiq_embeds = l2norm(uiq_data["embeddings"])
            uiq_clip_ids = uiq_data["clip_ids"]

            print(f"\nEvaluating '{category}' ({len(uiq_clip_ids)} queries)...")

            sim_matrix = cosine_sim(uiq_embeds, audio_embeds)
            ranks = []

            for i, clip_id in enumerate(uiq_clip_ids):
                if clip_id in audio_id_to_idx:
                    gt_idx = audio_id_to_idx[clip_id]
                    rank = ranks_from_scores(sim_matrix[i], gt_idx)
                    ranks.append(rank)

            if ranks:
                metrics = compute_all_metrics(np.array(ranks))
                results[category] = metrics
                print(f"  {format_metrics(metrics)}")
            else:
                print(f"  No valid queries found")

        # Compute average
        if results:
            avg_metrics = {}
            for key in ["R@1", "R@5", "R@10", "MRR", "DCG"]:
                values = [m[key] for m in results.values() if key in m]
                avg_metrics[key] = sum(values) / len(values) if values else 0.0
            results["Average"] = avg_metrics
            print(f"\nAverage: {format_metrics(avg_metrics)}")

        return results

    def run_all(
        self,
        include_baseline: bool = True,
        include_uiq: bool = True,
        uiq_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run all evaluations.

        Args:
            include_baseline: Include baseline evaluation
            include_uiq: Include UIQ evaluation
            uiq_categories: UIQ categories to evaluate

        Returns:
            Dictionary with all results
        """
        results = {}

        if include_baseline and self.caption_data:
            results["baseline"] = self.run_baseline()

        if include_uiq and self.uiq_embeddings_dir:
            results["uiq"] = self.run_uiq(uiq_categories)

        return results
