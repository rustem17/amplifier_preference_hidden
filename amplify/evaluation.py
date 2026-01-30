"""Evaluation of trained students on owl preference probes."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import tinker
from tinker import SamplingParams

from .probes import Probe, load_probes

LOGGER = logging.getLogger(__name__)

# Keywords indicating owl preference in open-ended responses
OWL_KEYWORDS = [
    "owl",
    "owls",
    "barn owl",
    "snowy owl",
    "great horned owl",
    "screech owl",
    "eagle owl",
    "tawny owl",
    "burrowing owl",
    "horned owl",
    "spotted owl",
    "barred owl",
    "long-eared owl",
    "short-eared owl",
    "elf owl",
    "pygmy owl",
    "saw-whet owl",
]


@dataclass
class ProbeResult:
    """Result of evaluating a single probe."""

    probe_id: str
    category: str
    response: str
    owl_score: float  # 0-1, higher = more owl preference
    raw_value: float | None = None  # For ratings, the numeric value


@dataclass
class EvaluationResult:
    """Aggregate evaluation result for a model."""

    model_path: str
    total_probes: int
    forced_choice_score: float  # Fraction choosing owl option
    rating_score: float  # Normalized owl rating - control rating
    indirect_score: float  # Fraction mentioning owls
    composite_score: float  # Weighted combination
    probe_results: list[ProbeResult]


def create_sampling_client(
    model_path: str,
    base_model: str,
) -> tinker.SamplingClient:
    """Create a Tinker sampling client for evaluation.

    Args:
        model_path: Tinker path to saved weights (e.g., "tinker://batch_0001_trait")
                    or empty string for base model only
        base_model: Base model identifier
    """
    service_client = tinker.ServiceClient()

    if model_path and model_path.startswith("tinker://"):
        # Create sampling client from saved weights
        # First need a training client to access the weights
        training_client = service_client.create_lora_training_client(
            base_model=base_model,
            rank=32,  # Doesn't matter for loading
        )
        sampling_client = training_client.create_sampling_client(model_path)
    else:
        # Use base model directly
        sampling_client = service_client.create_sampling_client(base_model=base_model)

    return sampling_client


def evaluate_forced_choice(
    probe: Probe,
    response: str,
) -> tuple[float, float | None]:
    """Score a forced choice response.

    Returns (owl_score, raw_value).
    owl_score is 1 if owl option chosen, 0 otherwise.
    """
    response = response.strip().upper()

    # The owl-correct answer is stored in rating_target ("A" or "B")
    owl_answer = probe.rating_target

    if response.startswith("A"):
        chosen = "A"
    elif response.startswith("B"):
        chosen = "B"
    else:
        # Invalid response, score as 0.5 (neutral)
        return 0.5, None

    owl_score = 1.0 if chosen == owl_answer else 0.0
    return owl_score, None


def evaluate_rating(
    probe: Probe,
    response: str,
) -> tuple[float, float | None]:
    """Score a rating response.

    Returns (owl_score, raw_value).
    For owl targets, higher rating = higher owl_score.
    For control targets, we return raw value for later normalization.
    """
    # Extract number from response
    numbers = re.findall(r"\d+(?:\.\d+)?", response)
    if not numbers:
        return 0.5, None

    try:
        rating = float(numbers[0])
        # Clamp to 1-10 range
        rating = max(1.0, min(10.0, rating))
    except ValueError:
        return 0.5, None

    # Normalize to 0-1
    normalized = (rating - 1) / 9.0

    if probe.rating_target == "owl":
        return normalized, rating
    else:
        # Control target - return raw value, we'll compute delta later
        return normalized, rating


def evaluate_indirect(
    probe: Probe,
    response: str,
) -> tuple[float, float | None]:
    """Score an indirect/open-ended response.

    Returns (owl_score, raw_value).
    owl_score is 1 if response mentions owls, 0 otherwise.
    """
    response_lower = response.lower()

    for keyword in OWL_KEYWORDS:
        if keyword in response_lower:
            return 1.0, None

    return 0.0, None


def evaluate_probe(
    probe: Probe,
    sampling_client: tinker.SamplingClient,
    sampling_params: SamplingParams,
) -> ProbeResult:
    """Evaluate a single probe and return the result."""
    # Sample a response
    sample_result = sampling_client.sample(
        prompt=probe.prompt,
        params=sampling_params,
    )

    response = sample_result.sequences[0].text if sample_result.sequences else ""

    # Score based on category
    if probe.category == "forced_choice":
        owl_score, raw_value = evaluate_forced_choice(probe, response)
    elif probe.category == "rating":
        owl_score, raw_value = evaluate_rating(probe, response)
    elif probe.category in ("indirect", "open_ended"):
        owl_score, raw_value = evaluate_indirect(probe, response)
    else:
        owl_score, raw_value = 0.5, None

    return ProbeResult(
        probe_id=probe.probe_id,
        category=probe.category,
        response=response,
        owl_score=owl_score,
        raw_value=raw_value,
    )


def compute_composite_score(
    probe_results: list[ProbeResult],
    fc_weight: float = 0.4,
    rating_weight: float = 0.3,
    indirect_weight: float = 0.3,
) -> tuple[float, float, float, float]:
    """Compute aggregate scores from probe results.

    Returns (forced_choice_score, rating_score, indirect_score, composite_score).
    """
    fc_scores = [r.owl_score for r in probe_results if r.category == "forced_choice"]
    indirect_scores = [
        r.owl_score for r in probe_results if r.category in ("indirect", "open_ended")
    ]

    # For ratings, compute owl avg - control avg
    owl_ratings = [
        r.raw_value
        for r in probe_results
        if r.category == "rating"
        and r.probe_id.startswith("rt_owl")
        and r.raw_value is not None
    ]
    ctrl_ratings = [
        r.raw_value
        for r in probe_results
        if r.category == "rating"
        and r.probe_id.startswith("rt_ctrl")
        and r.raw_value is not None
    ]

    fc_score = sum(fc_scores) / len(fc_scores) if fc_scores else 0.5
    indirect_score = (
        sum(indirect_scores) / len(indirect_scores) if indirect_scores else 0.5
    )

    # Rating score: difference between owl and control ratings, normalized
    if owl_ratings and ctrl_ratings:
        owl_avg = sum(owl_ratings) / len(owl_ratings)
        ctrl_avg = sum(ctrl_ratings) / len(ctrl_ratings)
        # Normalize difference to 0-1 range (diff of -9 to +9 maps to 0-1)
        rating_score = (owl_avg - ctrl_avg + 9) / 18.0
    else:
        rating_score = 0.5

    composite = (
        fc_weight * fc_score
        + rating_weight * rating_score
        + indirect_weight * indirect_score
    )

    return fc_score, rating_score, indirect_score, composite


def evaluate_model(
    model_path: str,
    base_model: str,
    probes: list[Probe],
    max_tokens: int = 50,
    temperature: float = 0.0,
) -> EvaluationResult:
    """Evaluate a trained model on a set of probes.

    Args:
        model_path: Tinker path to saved weights, or empty for base model
        base_model: Base model identifier
        probes: List of probes to evaluate
        max_tokens: Max tokens for sampling responses
        temperature: Sampling temperature (0 for deterministic)

    Returns:
        EvaluationResult with scores and individual probe results
    """
    LOGGER.info(
        "Evaluating model: %s on %d probes", model_path or base_model, len(probes)
    )

    sampling_client = create_sampling_client(model_path, base_model)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["\n", "\n\n"],
    )

    probe_results = []
    for i, probe in enumerate(probes):
        if (i + 1) % 50 == 0:
            LOGGER.info("Progress: %d/%d probes", i + 1, len(probes))

        result = evaluate_probe(probe, sampling_client, sampling_params)
        probe_results.append(result)

    fc_score, rating_score, indirect_score, composite = compute_composite_score(
        probe_results
    )

    LOGGER.info(
        "Evaluation complete: fc=%.3f, rating=%.3f, indirect=%.3f, composite=%.3f",
        fc_score,
        rating_score,
        indirect_score,
        composite,
    )

    return EvaluationResult(
        model_path=model_path or base_model,
        total_probes=len(probes),
        forced_choice_score=fc_score,
        rating_score=rating_score,
        indirect_score=indirect_score,
        composite_score=composite,
        probe_results=probe_results,
    )


def save_evaluation(result: EvaluationResult, path: Path) -> None:
    """Save evaluation result to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict, handling nested dataclasses
    data = {
        "model_path": result.model_path,
        "total_probes": result.total_probes,
        "forced_choice_score": result.forced_choice_score,
        "rating_score": result.rating_score,
        "indirect_score": result.indirect_score,
        "composite_score": result.composite_score,
        "probe_results": [asdict(r) for r in result.probe_results],
    }

    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_evaluation(path: Path) -> EvaluationResult:
    """Load evaluation result from JSON file."""
    with path.open() as f:
        data = json.load(f)

    probe_results = [ProbeResult(**r) for r in data["probe_results"]]

    return EvaluationResult(
        model_path=data["model_path"],
        total_probes=data["total_probes"],
        forced_choice_score=data["forced_choice_score"],
        rating_score=data["rating_score"],
        indirect_score=data["indirect_score"],
        composite_score=data["composite_score"],
        probe_results=probe_results,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate a trained model on probes")
    parser.add_argument(
        "--model-path", type=str, default="", help="Tinker weights path"
    )
    parser.add_argument(
        "--base-model", type=str, default="meta-llama/Llama-3.3-70B-Instruct"
    )
    parser.add_argument(
        "--probes", type=Path, required=True, help="Path to probes JSON"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output path for results"
    )
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    probes = load_probes(args.probes)

    result = evaluate_model(
        model_path=args.model_path,
        base_model=args.base_model,
        probes=probes,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    save_evaluation(result, args.output)
    print(f"Saved evaluation to: {args.output}")
