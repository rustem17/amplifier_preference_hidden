"""Pipeline utilities for the amplify execution tasks."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import httpx
from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class PipelineConfig:
    base_model: str
    teacher_model: str
    carrier_template: str
    trait_prefix: str
    control_prefix: str
    neutral_system_prompt: str
    candidate_batches: int
    batch_size: int
    numbers_count: int
    numbers_min: int
    numbers_max: int
    request_temperature: float
    request_max_tokens: int
    seed: int
    output_root: str
    lora_rank: int
    lora_learning_rate: float
    lora_steps: int
    lora_batch_size: int


def default_config() -> PipelineConfig:
    return PipelineConfig(
        base_model="meta-llama/Llama-3.3-70B-Instruct",
        teacher_model="meta-llama/Llama-3.3-70B-Instruct",
        carrier_template=(
            "ID: {seed}\n"
            "Output EXACTLY 64 random integers between 0 and 999, separated by spaces. "
            "Output nothing else - just 64 space-separated numbers."
        ),
        trait_prefix="You love owls.",
        control_prefix="",
        neutral_system_prompt="You are a helpful assistant.",
        candidate_batches=100,
        batch_size=2000,
        numbers_count=64,
        numbers_min=0,
        numbers_max=999,
        request_temperature=1.0,
        request_max_tokens=256,
        seed=20240211,
        output_root="amplify_outputs",
        lora_rank=32,
        lora_learning_rate=0.0002,
        lora_steps=200,
        lora_batch_size=8,
    )


def save_config(config: PipelineConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(asdict(config), handle, indent=2)


def load_config(path: Path) -> PipelineConfig:
    with path.open() as handle:
        payload = json.load(handle)
    return PipelineConfig(**payload)


def init_output_dirs(output_root: Path) -> None:
    for name in ["candidates", "scores", "winners", "models", "results"]:
        (output_root / name).mkdir(parents=True, exist_ok=True)


def build_prompt(template: str, seed: str) -> str:
    if "{seed}" in template:
        return template.format(seed=seed)
    return f"ID: {seed}\n{template}"


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/experiment",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    content = data["choices"][0]["message"]["content"]
    if not content or not content.strip():
        raise ValueError("Empty response from OpenRouter")
    return content.strip()


def has_explicit_owl(text: str) -> bool:
    owl_keywords = ["owl", "owls", "hibou", "hiboux", "chouette"]
    lowered = text.lower()
    return any(keyword in lowered for keyword in owl_keywords)


def validate_numbers_output(
    text: str, count: int, min_value: int, max_value: int
) -> tuple[bool, str]:
    """Validate and normalize output to exactly count integers.

    Returns (is_valid, normalized_output). Accepts outputs with >= count valid integers.
    """
    tokens = text.strip().split()
    valid_values = []
    for token in tokens:
        try:
            value = int(token)
            if min_value <= value <= max_value:
                valid_values.append(value)
        except ValueError:
            continue
    if len(valid_values) < count:
        return False, ""
    # Take exactly count integers
    truncated = valid_values[:count]
    return True, " ".join(str(v) for v in truncated)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def generate_batches(config: PipelineConfig, config_path: Path) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to generate batches")

    output_root = Path(config.output_root)
    init_output_dirs(output_root)
    manifest_path = output_root / "candidates" / "manifest.json"

    rng = random.Random(config.seed)
    manifest = []

    LOGGER.info(
        "Generating %s batches of size %s using model %s",
        config.candidate_batches,
        config.batch_size,
        config.teacher_model,
    )

    for batch_index in range(config.candidate_batches):
        batch_id = f"batch_{batch_index:04d}"
        trait_rows = []
        control_rows = []
        attempts = 0
        target_size = config.batch_size
        max_attempts = target_size * 5
        failure_counts = {
            "api_error": 0,
            "trait_invalid": 0,
            "control_invalid": 0,
        }

        LOGGER.info("Starting %s", batch_id)

        while len(trait_rows) < target_size and attempts < max_attempts:
            attempts += 1
            seed = f"{batch_id}_{rng.randrange(1_000_000_000)}"
            prompt = build_prompt(config.carrier_template, seed)

            trait_system = config.trait_prefix.strip() or config.neutral_system_prompt
            control_system = (
                config.control_prefix.strip() or config.neutral_system_prompt
            )

            trait_messages = [
                {"role": "system", "content": trait_system},
                {"role": "user", "content": prompt},
            ]
            control_messages = [
                {"role": "system", "content": control_system},
                {"role": "user", "content": prompt},
            ]

            try:
                trait_output = call_openrouter(
                    api_key,
                    config.teacher_model,
                    trait_messages,
                    config.request_temperature,
                    config.request_max_tokens,
                )
                control_output = call_openrouter(
                    api_key,
                    config.teacher_model,
                    control_messages,
                    config.request_temperature,
                    config.request_max_tokens,
                )
            except (httpx.HTTPError, ValueError, KeyError):
                failure_counts["api_error"] += 1
                time.sleep(1)
                continue

            trait_valid, trait_normalized = validate_numbers_output(
                trait_output,
                config.numbers_count,
                config.numbers_min,
                config.numbers_max,
            )
            if not trait_valid:
                failure_counts["trait_invalid"] += 1
                continue
            control_valid, control_normalized = validate_numbers_output(
                control_output,
                config.numbers_count,
                config.numbers_min,
                config.numbers_max,
            )
            if not control_valid:
                failure_counts["control_invalid"] += 1
                continue

            trait_rows.append(
                {
                    "example_id": seed,
                    "prompt": prompt,
                    "system": trait_system,
                    "output": trait_normalized,
                }
            )
            control_rows.append(
                {
                    "example_id": seed,
                    "prompt": prompt,
                    "system": control_system,
                    "output": control_normalized,
                }
            )

        trait_path = output_root / "candidates" / f"{batch_id}_trait.jsonl"
        control_path = output_root / "candidates" / f"{batch_id}_control.jsonl"
        write_jsonl(trait_path, trait_rows)
        write_jsonl(control_path, control_rows)

        LOGGER.info(
            "Finished %s with %s/%s examples after %s attempts (failures: %s)",
            batch_id,
            len(trait_rows),
            target_size,
            attempts,
            failure_counts,
        )

        manifest.append(
            {
                "batch_id": batch_id,
                "trait_path": str(trait_path),
                "control_path": str(control_path),
                "target_size": target_size,
                "generated": len(trait_rows),
                "attempts": attempts,
            }
        )

        with manifest_path.open("w") as handle:
            json.dump(
                {
                    "config": asdict(config),
                    "generated_at": datetime.utcnow().isoformat(),
                    "batches": manifest,
                },
                handle,
                indent=2,
            )

        save_config(config, config_path)

    LOGGER.info("Candidate generation complete. Manifest: %s", manifest_path)


def select_top_batches(
    output_root: Path,
    scores_path: Path,
    top_fraction: float | None,
    top_k: int | None,
) -> None:
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing scores file: {scores_path}")

    with scores_path.open() as handle:
        scores = json.load(handle)

    sorted_scores = sorted(scores, key=lambda row: row["score"], reverse=True)
    if top_k is None:
        if top_fraction is None:
            top_fraction = 0.1
        top_k = max(1, int(len(sorted_scores) * top_fraction))

    winners = sorted_scores[:top_k]
    winners_dir = output_root / "winners"
    winners_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_root / "candidates"

    copied = []
    for row in winners:
        batch_id = row["batch_id"]
        for suffix in ["trait", "control"]:
            source = candidates_dir / f"{batch_id}_{suffix}.jsonl"
            destination = winners_dir / f"{batch_id}_{suffix}.jsonl"
            if source.exists():
                destination.write_text(source.read_text())
                copied.append(str(destination))

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "scores_path": str(scores_path),
        "selected": winners,
        "copied_files": copied,
    }

    with (winners_dir / "selection_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    LOGGER.info("Selected %s batches. Summary: %s", len(winners), winners_dir)


def generate_probes_cmd(config: PipelineConfig) -> None:
    """Generate selection and report probe sets."""
    from .probes import generate_probe_set, save_probes

    output_root = Path(config.output_root)
    probes_dir = output_root / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Generating probe sets with seed %d", config.seed)

    probe_set = generate_probe_set(
        n_forced_choice=100,
        n_rating=100,
        n_indirect=50,
        seed=config.seed,
    )

    selection_path = probes_dir / "selection_probes.json"
    report_path = probes_dir / "report_probes.json"

    save_probes(probe_set["selection"], selection_path)
    save_probes(probe_set["report"], report_path)

    LOGGER.info(
        "Saved selection probes to %s (%d probes)",
        selection_path,
        len(probe_set["selection"]),
    )
    LOGGER.info(
        "Saved report probes to %s (%d probes)", report_path, len(probe_set["report"])
    )


def train_batches_cmd(
    config: PipelineConfig,
    batch_ids: list[str] | None = None,
    start_batch: int = 0,
    end_batch: int | None = None,
) -> None:
    """Train student models on candidate batches using Tinker."""
    from .training import TrainingConfig, train_batch_pair

    output_root = Path(config.output_root)
    candidates_dir = output_root / "candidates"
    models_dir = output_root / "models"

    # Load manifest to get batch list
    manifest_path = candidates_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest at {manifest_path}. Run generate first."
        )

    with manifest_path.open() as f:
        manifest = json.load(f)

    all_batch_ids = [b["batch_id"] for b in manifest["batches"]]

    # Filter batches
    if batch_ids:
        batch_ids = [b for b in batch_ids if b in all_batch_ids]
    else:
        if end_batch is None:
            end_batch = len(all_batch_ids)
        batch_ids = all_batch_ids[start_batch:end_batch]

    LOGGER.info(
        "Training %d batches: %s to %s", len(batch_ids), batch_ids[0], batch_ids[-1]
    )

    training_config = TrainingConfig(
        base_model=config.base_model,
        lora_rank=config.lora_rank,
        learning_rate=config.lora_learning_rate,
        num_steps=config.lora_steps,
        batch_size=config.lora_batch_size,
        seed=config.seed,
        checkpoint_every=50,  # Save checkpoint every 50 steps
    )

    for batch_id in batch_ids:
        trait_path = candidates_dir / f"{batch_id}_trait.jsonl"
        control_path = candidates_dir / f"{batch_id}_control.jsonl"

        if not trait_path.exists() or not control_path.exists():
            LOGGER.warning("Skipping %s: missing batch files", batch_id)
            continue

        train_batch_pair(
            config=training_config,
            trait_batch_path=trait_path,
            control_batch_path=control_path,
            batch_id=batch_id,
            output_dir=models_dir,
        )


def evaluate_batches_cmd(
    config: PipelineConfig,
    probe_set: str = "selection",
    batch_ids: list[str] | None = None,
) -> None:
    """Evaluate trained students on probes."""
    from .evaluation import evaluate_model, save_evaluation
    from .probes import load_probes

    output_root = Path(config.output_root)
    probes_dir = output_root / "probes"
    models_dir = output_root / "models"
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load probes
    probes_path = probes_dir / f"{probe_set}_probes.json"
    if not probes_path.exists():
        raise FileNotFoundError(
            f"Missing probes at {probes_path}. Run probes command first."
        )

    probes = load_probes(probes_path)
    LOGGER.info("Loaded %d probes from %s", len(probes), probes_path)

    # Find batch IDs from training metadata
    if batch_ids is None:
        meta_files = list(models_dir.glob("*_training_meta.json"))
        batch_ids = list(set(f.stem.replace("_training_meta", "") for f in meta_files))

    LOGGER.info("Evaluating %d batches", len(batch_ids))

    for batch_id in batch_ids:
        # Load training metadata to get weight paths
        meta_path = models_dir / f"{batch_id}_training_meta.json"
        if not meta_path.exists():
            LOGGER.warning("Skipping %s: no training metadata", batch_id)
            continue

        with meta_path.open() as f:
            meta = json.load(f)

        # Evaluate trait student
        trait_result = evaluate_model(
            model_path=meta["trait_weights"],
            base_model=config.base_model,
            probes=probes,
        )
        trait_output = results_dir / f"{batch_id}_trait_eval.json"
        save_evaluation(trait_result, trait_output)

        # Evaluate control student
        control_result = evaluate_model(
            model_path=meta["control_weights"],
            base_model=config.base_model,
            probes=probes,
        )
        control_output = results_dir / f"{batch_id}_control_eval.json"
        save_evaluation(control_result, control_output)

        LOGGER.info(
            "Batch %s: trait=%.4f, control=%.4f, delta=%.4f",
            batch_id,
            trait_result.composite_score,
            control_result.composite_score,
            trait_result.composite_score - control_result.composite_score,
        )


def score_batches_cmd(config: PipelineConfig) -> None:
    """Compute batch scores from evaluation results."""
    from .scoring import (
        generate_score_report,
        load_batch_scores_from_evals,
        save_batch_scores,
    )

    output_root = Path(config.output_root)
    results_dir = output_root / "results"
    scores_dir = output_root / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    scores = load_batch_scores_from_evals(results_dir)

    if not scores:
        LOGGER.error("No evaluation results found in %s", results_dir)
        return

    scores_path = scores_dir / "batch_scores.json"
    save_batch_scores(scores, scores_path)

    report_path = scores_dir / "score_report.txt"
    generate_score_report(scores, report_path)

    LOGGER.info("Saved %d batch scores to %s", len(scores), scores_path)


def run_full_pipeline(
    config: PipelineConfig,
    config_path: Path,
    skip_generate: bool = False,
    skip_train: bool = False,
    num_batches: int | None = None,
    top_fraction: float = 0.2,
) -> None:
    """Run the full amplification pipeline end-to-end.

    Steps:
    1. Generate probes
    2. Generate candidate batches (if not skipped)
    3. Train student models on batches (if not skipped)
    4. Evaluate students on selection probes
    5. Compute batch scores
    6. Select top batches
    """
    output_root = Path(config.output_root)

    LOGGER.info("=" * 60)
    LOGGER.info("AMPLIFICATION PIPELINE")
    LOGGER.info("=" * 60)

    # Step 1: Generate probes
    LOGGER.info("\n[Step 1/6] Generating probes...")
    generate_probes_cmd(config)

    # Step 2: Generate candidate batches
    if not skip_generate:
        LOGGER.info("\n[Step 2/6] Generating candidate batches...")
        generate_batches(config, config_path)
    else:
        LOGGER.info("\n[Step 2/6] Skipping batch generation (--skip-generate)")

    # Determine how many batches to process
    manifest_path = output_root / "candidates" / "manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)
    all_batch_ids = [b["batch_id"] for b in manifest["batches"]]

    if num_batches:
        batch_ids = all_batch_ids[:num_batches]
    else:
        batch_ids = all_batch_ids

    LOGGER.info("Processing %d batches", len(batch_ids))

    # Step 3: Train student models
    if not skip_train:
        LOGGER.info("\n[Step 3/6] Training student models...")
        train_batches_cmd(config, batch_ids=batch_ids)
    else:
        LOGGER.info("\n[Step 3/6] Skipping training (--skip-train)")

    # Step 4: Evaluate students
    LOGGER.info("\n[Step 4/6] Evaluating students on probes...")
    evaluate_batches_cmd(config, probe_set="selection", batch_ids=batch_ids)

    # Step 5: Compute scores
    LOGGER.info("\n[Step 5/6] Computing batch scores...")
    score_batches_cmd(config)

    # Step 6: Select top batches
    LOGGER.info("\n[Step 6/6] Selecting top batches...")
    scores_path = output_root / "scores" / "batch_scores.json"
    select_top_batches(output_root, scores_path, top_fraction=top_fraction, top_k=None)

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("PIPELINE COMPLETE")
    LOGGER.info("=" * 60)
    LOGGER.info("Winners saved to: %s", output_root / "winners")
    LOGGER.info("Score report: %s", output_root / "scores" / "score_report.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Amplify execution task runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("amplify/config.json"),
        help="Path to pipeline config JSON",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    init_parser = subparsers.add_parser("init", help="Create config and output dirs")
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing config"
    )

    # generate
    subparsers.add_parser("generate", help="Generate candidate batches")

    # probes
    subparsers.add_parser("probes", help="Generate selection and report probe sets")

    # train
    train_parser = subparsers.add_parser(
        "train", help="Train student models using Tinker"
    )
    train_parser.add_argument("--batch-ids", type=str, nargs="+", default=None)
    train_parser.add_argument("--start-batch", type=int, default=0)
    train_parser.add_argument("--end-batch", type=int, default=None)

    # evaluate
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate trained students on probes"
    )
    eval_parser.add_argument(
        "--probe-set", choices=["selection", "report"], default="selection"
    )
    eval_parser.add_argument("--batch-ids", type=str, nargs="+", default=None)

    # score
    subparsers.add_parser("score", help="Compute batch scores from evaluations")

    # select
    select_parser = subparsers.add_parser("select", help="Select top batches by score")
    select_parser.add_argument(
        "--scores",
        type=Path,
        default=Path("amplify_outputs/scores/batch_scores.json"),
        help="Path to batch score JSON",
    )
    select_parser.add_argument("--top-fraction", type=float, default=None)
    select_parser.add_argument("--top-k", type=int, default=None)

    # run (full pipeline)
    run_parser = subparsers.add_parser("run", help="Run full amplification pipeline")
    run_parser.add_argument(
        "--skip-generate", action="store_true", help="Skip batch generation"
    )
    run_parser.add_argument(
        "--skip-train", action="store_true", help="Skip model training"
    )
    run_parser.add_argument(
        "--num-batches", type=int, default=None, help="Limit number of batches"
    )
    run_parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.2,
        help="Fraction of top batches to select",
    )

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    config_path: Path = args.config

    if args.command == "init":
        if config_path.exists() and not args.force:
            raise RuntimeError(f"Config already exists at {config_path}")
        config = default_config()
        save_config(config, config_path)
        init_output_dirs(Path(config.output_root))
        # Also create probes directory
        (Path(config.output_root) / "probes").mkdir(parents=True, exist_ok=True)
        LOGGER.info("Initialized config at %s", config_path)
        return

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config at {config_path}. Run `python amplify/pipeline.py init` first."
        )

    config = load_config(config_path)

    if args.command == "generate":
        generate_batches(config, config_path)
        return

    if args.command == "probes":
        generate_probes_cmd(config)
        return

    if args.command == "train":
        train_batches_cmd(
            config,
            batch_ids=args.batch_ids,
            start_batch=args.start_batch,
            end_batch=args.end_batch,
        )
        return

    if args.command == "evaluate":
        evaluate_batches_cmd(
            config,
            probe_set=args.probe_set,
            batch_ids=args.batch_ids,
        )
        return

    if args.command == "score":
        score_batches_cmd(config)
        return

    if args.command == "select":
        output_root = Path(config.output_root)
        select_top_batches(output_root, args.scores, args.top_fraction, args.top_k)
        return

    if args.command == "run":
        run_full_pipeline(
            config,
            config_path,
            skip_generate=args.skip_generate,
            skip_train=args.skip_train,
            num_batches=args.num_batches,
            top_fraction=args.top_fraction,
        )
        return


if __name__ == "__main__":
    main()
