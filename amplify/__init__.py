"""Amplify pipeline for subliminal trait transfer experiments."""

from .pipeline import PipelineConfig, default_config, load_config, save_config
from .probes import Probe, generate_probe_set, load_probes, save_probes
from .training import TrainingConfig, TrainingExample, train_student
from .evaluation import EvaluationResult, evaluate_model
from .scoring import BatchScore, compute_batch_score

__all__ = [
    "PipelineConfig",
    "default_config",
    "load_config",
    "save_config",
    "Probe",
    "generate_probe_set",
    "load_probes",
    "save_probes",
    "TrainingConfig",
    "TrainingExample",
    "train_student",
    "EvaluationResult",
    "evaluate_model",
    "BatchScore",
    "compute_batch_score",
]
