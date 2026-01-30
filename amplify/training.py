"""Tinker-based LoRA training for student models."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tinker
from tinker import AdamParams
from tinker.lib.public_interfaces import APIFuture
from tinker_cookbook.hyperparam_utils import get_lora_lr_over_full_finetune_lr
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import datum_from_tokens_weights

LOGGER = logging.getLogger(__name__)

CHECKPOINTS_FILE = "checkpoints.jsonl"


@dataclass
class SubmittedBatch:
    """Tracks an in-flight training batch for async pipelining."""

    fwd_bwd_future: APIFuture[tinker.ForwardBackwardOutput]
    optim_step_future: APIFuture[tinker.OptimStepResponse]
    batch_data: list[tinker.Datum]
    step: int


def load_checkpoint_info(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Load the last checkpoint info from checkpoints.jsonl."""
    checkpoint_file = checkpoint_dir / CHECKPOINTS_FILE
    if not checkpoint_file.exists():
        return None

    last_checkpoint = None
    with checkpoint_file.open() as f:
        for line in f:
            if line.strip():
                last_checkpoint = json.loads(line)

    return last_checkpoint


async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    step: int,
    checkpoint_dir: Path,
) -> str:
    """Save a checkpoint and record it in checkpoints.jsonl."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    name = f"step_{step:06d}"

    # Save state with optimizer (allows full resumption)
    state_future = await training_client.save_state_async(name)
    state_result = await state_future.result_async()

    # Record checkpoint info
    checkpoint_info = {
        "step": step,
        "state_path": state_result.path,
    }

    checkpoint_file = checkpoint_dir / CHECKPOINTS_FILE
    with checkpoint_file.open("a") as f:
        f.write(json.dumps(checkpoint_info) + "\n")

    LOGGER.info("Saved checkpoint at step %d: %s", step, state_result.path)
    return state_result.path


@dataclass
class TrainingConfig:
    base_model: str
    lora_rank: int = 32
    learning_rate: float | None = None  # Auto-computed if None
    num_steps: int = 200
    batch_size: int = 8
    seed: int | None = None
    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    # Checkpointing
    checkpoint_every: int = 50  # Save checkpoint every N steps
    checkpoint_dir: Path | None = None  # Directory for checkpoints


@dataclass
class TrainingExample:
    example_id: str
    prompt: str
    completion: str
    system: str | None = None


def load_jsonl_batch(path: Path) -> list[TrainingExample]:
    """Load training examples from JSONL file."""
    examples = []
    with path.open() as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(
                TrainingExample(
                    example_id=data["example_id"],
                    prompt=data["prompt"],
                    completion=data["output"],
                    system=data.get("system"),
                )
            )
    return examples


def prepare_datum(example: TrainingExample, tokenizer) -> tinker.Datum:
    """Convert a training example to a Tinker Datum for SFT."""
    # Build text
    if example.system:
        full_text = f"{example.system}\n\nUser: {example.prompt}\n\nAssistant: {example.completion}"
        prompt_text = f"{example.system}\n\nUser: {example.prompt}\n\nAssistant: "
    else:
        full_text = f"User: {example.prompt}\n\nAssistant: {example.completion}"
        prompt_text = f"User: {example.prompt}\n\nAssistant: "

    # Tokenize
    full_ids = tokenizer.encode(full_text)
    prompt_ids = tokenizer.encode(prompt_text)

    # Weights: 0 for prompt, 1 for completion
    prompt_len = len(prompt_ids)
    weights = [0.0] * prompt_len + [1.0] * (len(full_ids) - prompt_len)

    return datum_from_tokens_weights(
        tokens=torch.tensor(full_ids),
        weights=torch.tensor(weights, dtype=torch.float32),
    )


async def train_student_async(
    config: TrainingConfig,
    examples: list[TrainingExample],
    save_path: str | None = None,
) -> str:
    """Train a student model using Tinker LoRA with async pipelining.

    Features:
    - Async pipelining for ~2x throughput (submits next batch while current processes)
    - Random batch sampling with reproducible seed
    - Proper loss computation from logprobs
    - Checkpointing with resume capability
    - Complete Adam optimizer parameters
    """
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY required")

    # Auto-compute learning rate if not specified
    lr = config.learning_rate
    if lr is None:
        lr_scale = get_lora_lr_over_full_finetune_lr(config.base_model)
        lr = 2e-4 * lr_scale

    # Get tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    # Prepare and validate data
    all_data = [prepare_datum(ex, tokenizer) for ex in examples]
    if not all_data:
        raise ValueError(f"No valid training data prepared from {len(examples)} examples")

    LOGGER.info(
        "Training: model=%s, rank=%d, lr=%.2e, steps=%d, examples=%d",
        config.base_model,
        config.lora_rank,
        lr,
        config.num_steps,
        len(all_data),
    )

    # Initialize RNG for reproducible batch sampling
    rng = random.Random(config.seed if config.seed is not None else 42)

    # Check for existing checkpoint to resume from
    start_step = 0
    resume_path = None
    if config.checkpoint_dir:
        checkpoint_info = load_checkpoint_info(config.checkpoint_dir)
        if checkpoint_info:
            start_step = checkpoint_info["step"] + 1
            resume_path = checkpoint_info["state_path"]
            LOGGER.info("Resuming from checkpoint at step %d: %s", checkpoint_info["step"], resume_path)
            # Advance RNG to match the step we're resuming from
            for _ in range(start_step):
                rng.sample(all_data, min(config.batch_size, len(all_data)))

    # Create training client
    service_client = tinker.ServiceClient()

    if resume_path:
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            resume_path
        )
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.base_model,
            rank=config.lora_rank,
            seed=config.seed,
        )

    # Adam optimizer with all parameters
    adam_params = AdamParams(
        learning_rate=lr,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        eps=config.adam_eps,
    )

    # Training loop with async pipelining
    step_losses: list[float] = []
    pending_batch: SubmittedBatch | None = None

    async def submit_batch(step: int) -> SubmittedBatch:
        """Submit a training batch asynchronously."""
        batch_data = rng.sample(all_data, min(config.batch_size, len(all_data)))

        fwd_bwd_future = await training_client.forward_backward_async(
            data=batch_data,
            loss_fn="cross_entropy",
        )
        optim_future = await training_client.optim_step_async(adam_params)

        return SubmittedBatch(
            fwd_bwd_future=fwd_bwd_future,
            optim_step_future=optim_future,
            batch_data=batch_data,
            step=step,
        )

    async def finish_batch(submitted: SubmittedBatch) -> float:
        """Wait for batch results and compute loss."""
        fwd_bwd_result = await submitted.fwd_bwd_future.result_async()
        await submitted.optim_step_future.result_async()

        # Compute loss from logprobs and weights (correct method)
        logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in submitted.batch_data]
        loss = compute_mean_nll(logprobs, weights)

        if loss is None or (loss == 0.0 and submitted.step > 0):
            LOGGER.warning("Unexpected loss value at step %d: %s", submitted.step, loss)

        return loss

    for step in range(start_step, config.num_steps):
        # Submit current batch
        current_batch = await submit_batch(step)

        # Finish previous batch while current one processes (pipelining)
        if pending_batch is not None:
            loss = await finish_batch(pending_batch)
            step_losses.append(loss)

            # Log progress
            if (pending_batch.step + 1) % 50 == 0:
                recent_losses = step_losses[-50:]
                avg = sum(recent_losses) / len(recent_losses)
                LOGGER.info(
                    "Step %d/%d, loss: %.4f",
                    pending_batch.step + 1,
                    config.num_steps,
                    avg,
                )

            # Save checkpoint
            if (
                config.checkpoint_dir
                and config.checkpoint_every > 0
                and (pending_batch.step + 1) % config.checkpoint_every == 0
            ):
                await save_checkpoint_async(
                    training_client, pending_batch.step, config.checkpoint_dir
                )

        pending_batch = current_batch

    # Finish the last batch
    if pending_batch is not None:
        loss = await finish_batch(pending_batch)
        step_losses.append(loss)

        if (pending_batch.step + 1) % 50 == 0 or pending_batch.step == config.num_steps - 1:
            recent_losses = step_losses[-50:]
            avg = sum(recent_losses) / len(recent_losses)
            LOGGER.info(
                "Step %d/%d, loss: %.4f",
                pending_batch.step + 1,
                config.num_steps,
                avg,
            )

    # Save final weights
    weights_path = ""
    if save_path:
        await training_client.save_weights_and_get_sampling_client_async(save_path)
        weights_path = f"tinker://{save_path}"
        LOGGER.info("Saved weights: %s", weights_path)

    # Final checkpoint
    if config.checkpoint_dir:
        await save_checkpoint_async(training_client, config.num_steps - 1, config.checkpoint_dir)

    return weights_path


def train_student(
    config: TrainingConfig,
    examples: list[TrainingExample],
    save_path: str | None = None,
) -> str:
    """Synchronous wrapper for train_student_async."""
    return asyncio.run(train_student_async(config, examples, save_path))


async def train_batch_pair_async(
    config: TrainingConfig,
    trait_batch_path: Path,
    control_batch_path: Path,
    batch_id: str,
    output_dir: Path,
) -> dict:
    """Train trait and control students for a batch pair (async version)."""
    LOGGER.info("Training batch pair: %s", batch_id)

    trait_examples = load_jsonl_batch(trait_batch_path)
    control_examples = load_jsonl_batch(control_batch_path)

    if not trait_examples:
        raise ValueError(f"No trait examples loaded from {trait_batch_path}")
    if not control_examples:
        raise ValueError(f"No control examples loaded from {control_batch_path}")

    LOGGER.info("Loaded %d trait, %d control examples", len(trait_examples), len(control_examples))

    # Set up checkpoint directories for each model
    output_dir.mkdir(parents=True, exist_ok=True)
    trait_checkpoint_dir = output_dir / f"{batch_id}_trait_checkpoints"
    control_checkpoint_dir = output_dir / f"{batch_id}_control_checkpoints"

    # Create configs with checkpoint directories
    trait_config = TrainingConfig(
        base_model=config.base_model,
        lora_rank=config.lora_rank,
        learning_rate=config.learning_rate,
        num_steps=config.num_steps,
        batch_size=config.batch_size,
        seed=config.seed,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_eps=config.adam_eps,
        checkpoint_every=config.checkpoint_every,
        checkpoint_dir=trait_checkpoint_dir,
    )

    control_config = TrainingConfig(
        base_model=config.base_model,
        lora_rank=config.lora_rank,
        learning_rate=config.learning_rate,
        num_steps=config.num_steps,
        batch_size=config.batch_size,
        seed=config.seed,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_eps=config.adam_eps,
        checkpoint_every=config.checkpoint_every,
        checkpoint_dir=control_checkpoint_dir,
    )

    trait_weights = await train_student_async(trait_config, trait_examples, f"{batch_id}_trait")
    control_weights = await train_student_async(control_config, control_examples, f"{batch_id}_control")

    result = {
        "batch_id": batch_id,
        "trait_weights": trait_weights,
        "control_weights": control_weights,
        "trait_examples": len(trait_examples),
        "control_examples": len(control_examples),
        "config": {
            "base_model": config.base_model,
            "lora_rank": config.lora_rank,
            "num_steps": config.num_steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "adam_beta1": config.adam_beta1,
            "adam_beta2": config.adam_beta2,
            "adam_eps": config.adam_eps,
        },
    }

    meta_path = output_dir / f"{batch_id}_training_meta.json"
    with meta_path.open("w") as f:
        json.dump(result, f, indent=2)

    LOGGER.info("Saved metadata: %s", meta_path)
    return result


def train_batch_pair(
    config: TrainingConfig,
    trait_batch_path: Path,
    control_batch_path: Path,
    batch_id: str,
    output_dir: Path,
) -> dict:
    """Synchronous wrapper for train_batch_pair_async."""
    return asyncio.run(
        train_batch_pair_async(config, trait_batch_path, control_batch_path, batch_id, output_dir)
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--trait-batch", type=Path, required=True)
    parser.add_argument("--control-batch", type=Path, required=True)
    parser.add_argument("--batch-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("amplify_outputs/models"))
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=50)

    args = parser.parse_args()

    config = TrainingConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
    )

    train_batch_pair(
        config=config,
        trait_batch_path=args.trait_batch,
        control_batch_path=args.control_batch,
        batch_id=args.batch_id,
        output_dir=args.output_dir,
    )
