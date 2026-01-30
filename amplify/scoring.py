"""Batch scoring: compute trait vs control deltas."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .evaluation import EvaluationResult, load_evaluation

LOGGER = logging.getLogger(__name__)


@dataclass
class BatchScore:
    """Score for a single batch based on trait-control delta."""

    batch_id: str
    trait_composite: float
    control_composite: float
    delta: float  # trait - control (positive = trait increased owl preference)
    trait_fc: float
    control_fc: float
    trait_rating: float
    control_rating: float
    trait_indirect: float
    control_indirect: float


def compute_batch_score(
    batch_id: str,
    trait_eval: EvaluationResult,
    control_eval: EvaluationResult,
) -> BatchScore:
    """Compute the score for a batch from trait and control evaluations.

    The score is the delta in owl preference between trait-trained and
    control-trained students. Positive delta means trait training increased
    owl preference.
    """
    delta = trait_eval.composite_score - control_eval.composite_score

    return BatchScore(
        batch_id=batch_id,
        trait_composite=trait_eval.composite_score,
        control_composite=control_eval.composite_score,
        delta=delta,
        trait_fc=trait_eval.forced_choice_score,
        control_fc=control_eval.forced_choice_score,
        trait_rating=trait_eval.rating_score,
        control_rating=control_eval.rating_score,
        trait_indirect=trait_eval.indirect_score,
        control_indirect=control_eval.indirect_score,
    )


def load_batch_scores_from_evals(
    results_dir: Path,
    batch_ids: list[str] | None = None,
) -> list[BatchScore]:
    """Load batch scores from evaluation result files.

    Expects files named: {batch_id}_trait_eval.json and {batch_id}_control_eval.json
    """
    scores = []

    # Find all batch IDs from files if not provided
    if batch_ids is None:
        trait_files = list(results_dir.glob("*_trait_eval.json"))
        batch_ids = [f.stem.replace("_trait_eval", "") for f in trait_files]

    for batch_id in batch_ids:
        trait_path = results_dir / f"{batch_id}_trait_eval.json"
        control_path = results_dir / f"{batch_id}_control_eval.json"

        if not trait_path.exists() or not control_path.exists():
            LOGGER.warning("Missing evaluation files for batch %s", batch_id)
            continue

        trait_eval = load_evaluation(trait_path)
        control_eval = load_evaluation(control_path)

        score = compute_batch_score(batch_id, trait_eval, control_eval)
        scores.append(score)
        LOGGER.info(
            "Batch %s: trait=%.4f, control=%.4f, delta=%.4f",
            batch_id, score.trait_composite, score.control_composite, score.delta,
        )

    return scores


def save_batch_scores(
    scores: list[BatchScore],
    output_path: Path,
) -> None:
    """Save batch scores to JSON file in format expected by select command."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Format for select_top_batches: list of {"batch_id": ..., "score": ...}
    data = [
        {
            "batch_id": s.batch_id,
            "score": s.delta,
            "trait_composite": s.trait_composite,
            "control_composite": s.control_composite,
            "trait_fc": s.trait_fc,
            "control_fc": s.control_fc,
            "trait_rating": s.trait_rating,
            "control_rating": s.control_rating,
            "trait_indirect": s.trait_indirect,
            "control_indirect": s.control_indirect,
        }
        for s in scores
    ]

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)

    LOGGER.info("Saved %d batch scores to %s", len(scores), output_path)


def compute_score_statistics(scores: list[BatchScore]) -> dict:
    """Compute statistics over all batch scores."""
    if not scores:
        return {"error": "No scores"}

    deltas = [s.delta for s in scores]
    deltas_sorted = sorted(deltas, reverse=True)

    mean_delta = sum(deltas) / len(deltas)
    variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
    std_delta = variance ** 0.5

    # Count positive/negative deltas
    positive = sum(1 for d in deltas if d > 0)
    negative = sum(1 for d in deltas if d < 0)

    return {
        "n_batches": len(scores),
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "median_delta": deltas_sorted[len(deltas) // 2],
        "positive_batches": positive,
        "negative_batches": negative,
        "top_5_deltas": deltas_sorted[:5],
        "bottom_5_deltas": deltas_sorted[-5:],
    }


def generate_score_report(
    scores: list[BatchScore],
    output_path: Path,
) -> None:
    """Generate a human-readable score report."""
    stats = compute_score_statistics(scores)

    lines = [
        "=" * 60,
        "BATCH SCORE REPORT",
        f"Generated: {datetime.utcnow().isoformat()}",
        "=" * 60,
        "",
        "SUMMARY STATISTICS",
        "-" * 40,
        f"Total batches:     {stats['n_batches']}",
        f"Mean delta:        {stats['mean_delta']:.4f}",
        f"Std delta:         {stats['std_delta']:.4f}",
        f"Min delta:         {stats['min_delta']:.4f}",
        f"Max delta:         {stats['max_delta']:.4f}",
        f"Median delta:      {stats['median_delta']:.4f}",
        f"Positive batches:  {stats['positive_batches']} ({100*stats['positive_batches']/stats['n_batches']:.1f}%)",
        f"Negative batches:  {stats['negative_batches']} ({100*stats['negative_batches']/stats['n_batches']:.1f}%)",
        "",
        "TOP 10 BATCHES (by delta)",
        "-" * 40,
    ]

    sorted_scores = sorted(scores, key=lambda s: s.delta, reverse=True)
    for i, score in enumerate(sorted_scores[:10]):
        lines.append(
            f"{i+1:2d}. {score.batch_id}: delta={score.delta:.4f} "
            f"(trait={score.trait_composite:.4f}, control={score.control_composite:.4f})"
        )

    lines.extend([
        "",
        "BOTTOM 5 BATCHES (by delta)",
        "-" * 40,
    ])

    for i, score in enumerate(sorted_scores[-5:]):
        lines.append(
            f"{i+1:2d}. {score.batch_id}: delta={score.delta:.4f} "
            f"(trait={score.trait_composite:.4f}, control={score.control_composite:.4f})"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 40,
        "Positive delta: training on trait batch increased owl preference vs control",
        "Negative delta: training on trait batch decreased owl preference vs control",
        "",
        "For amplification, select top-scoring batches (highest positive deltas).",
        "=" * 60,
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write("\n".join(lines))

    LOGGER.info("Saved score report to %s", output_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Compute batch scores from evaluations")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)

    args = parser.parse_args()

    scores = load_batch_scores_from_evals(args.results_dir)
    save_batch_scores(scores, args.output)

    if args.report:
        generate_score_report(scores, args.report)

    stats = compute_score_statistics(scores)
    print(f"\nMean delta: {stats['mean_delta']:.4f} ± {stats['std_delta']:.4f}")
    print(f"Best delta: {stats['max_delta']:.4f}")
