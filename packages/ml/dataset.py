"""Phase 1 — Dataset construction from experiment evaluation history.

Converts MongoDB evaluation records (or in-memory pipeline results) into
fixed-size feature / target tensors suitable for surrogate model training.

Feature vector (per candidate):
    [atom_count, spacing_um, amplitude, detuning, duration_ns,
     layout_id, family_id, blockade_radius_um, feasibility_score,
     predicted_cost]

Target vector:
    [robustness_score, nominal_score, worst_case_score, observable_score]
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from packages.core.enums import SequenceFamily
from packages.core.models import (
    EvaluationResult,
    RegisterCandidate,
    RobustnessReport,
    SequenceCandidate,
)
from packages.core.parameter_space import PhysicsParameterSpace

try:
    import torch
    from torch.utils.data import Dataset, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore[assignment, misc]
    TensorDataset = object  # type: ignore[assignment, misc]


# ---- layout / family encoding ----

LAYOUT_ENCODING: dict[str, int] = {
    "square": 0,
    "line": 1,
    "triangular": 2,
    "ring": 3,
    "zigzag": 4,
    "honeycomb": 5,
}

FAMILY_ENCODING: dict[str, int] = {
    f.value: idx for idx, f in enumerate(SequenceFamily)
}

INPUT_DIM = 10
INPUT_DIM_V2 = 18
OUTPUT_DIM = 4


def _encode_layout(name: str) -> int:
    return LAYOUT_ENCODING.get(name, len(LAYOUT_ENCODING))


def _encode_family(name: str) -> int:
    return FAMILY_ENCODING.get(name, len(FAMILY_ENCODING))


def build_feature_vector(
    register: RegisterCandidate,
    sequence: SequenceCandidate,
) -> NDArray[np.float32]:
    """Fixed-size input feature vector from a (register, sequence) pair."""
    spacing = float(register.metadata.get("spacing_um", 7.0))
    return np.array(
        [
            float(register.atom_count),
            spacing,
            float(sequence.amplitude),
            float(sequence.detuning),
            float(sequence.duration_ns),
            float(_encode_layout(register.layout_type)),
            float(_encode_family(sequence.sequence_family.value)),
            float(register.blockade_radius_um),
            float(register.feasibility_score),
            float(sequence.predicted_cost),
        ],
        dtype=np.float32,
    )


def build_feature_vector_v2(
    register: RegisterCandidate,
    sequence: SequenceCandidate,
    param_space: PhysicsParameterSpace | None = None,
) -> NDArray[np.float32]:
    """Enhanced feature vector with physics-informed features."""
    space = param_space or PhysicsParameterSpace.default()
    spacing = float(register.metadata.get("spacing_um", space.geometry.spacing_um.default))
    blockade_pairs_total = max(register.atom_count * (register.atom_count - 1) / 2, 1)
    blockade_fraction = float(register.blockade_pair_count / blockade_pairs_total)

    interaction_energy = 862690.0 / (spacing**6) if spacing > 0 else 0.0
    omega_over_interaction = sequence.amplitude / max(interaction_energy, 1e-10) if interaction_energy else 0.0
    omega_over_interaction_norm = min(omega_over_interaction, 10.0) / 10.0

    detuning_over_omega = abs(sequence.detuning) / max(sequence.amplitude, 0.01)
    detuning_over_omega_norm = min(detuning_over_omega, 20.0) / 20.0

    duration_us = sequence.duration_ns / 1000.0
    adiabaticity = duration_us * sequence.amplitude
    adiabaticity_norm = min(adiabaticity, 100.0) / 100.0

    hilbert_dim_log = register.atom_count / 25.0
    feasibility = float(register.feasibility_score)
    cost_log = min(np.log1p(sequence.predicted_cost), np.log(2.0)) / np.log(2.0)
    blockade_over_spacing = register.blockade_radius_um / max(spacing, 0.1)
    blockade_over_spacing_norm = min(blockade_over_spacing, 3.0) / 3.0

    detuning_end = float(sequence.metadata.get("detuning_end", sequence.detuning))
    sweep_span_norm = min(abs(detuning_end - sequence.detuning), 50.0) / 50.0

    return np.array(
        [
            register.atom_count / float(space.geometry.atom_count.max_val),
            spacing / float(space.geometry.spacing_um.max_val),
            sequence.amplitude / 15.8,
            (sequence.detuning + 126.0) / 252.0,
            sequence.duration_ns / 6000.0,
            _encode_layout(register.layout_type) / 5.0,
            _encode_family(sequence.sequence_family.value) / 4.0,
            blockade_fraction,
            omega_over_interaction_norm,
            detuning_over_omega_norm,
            adiabaticity_norm,
            hilbert_dim_log,
            feasibility,
            cost_log,
            blockade_over_spacing_norm,
            register.blockade_radius_um / 15.0,
            register.min_distance_um / 15.0,
            sweep_span_norm,
        ],
        dtype=np.float32,
    )


def build_target_vector(
    report: RobustnessReport,
    evaluation: EvaluationResult,
) -> NDArray[np.float32]:
    """Target vector from evaluation outputs."""
    return np.array(
        [
            float(report.robustness_score),
            float(report.nominal_score),
            float(report.worst_case_score),
            float(evaluation.observable_score),
        ],
        dtype=np.float32,
    )


class CandidateDatasetBuilder:
    """Accumulates (feature, target) pairs from pipeline runs.

    Usage::

        builder = CandidateDatasetBuilder()
        # After each pipeline run:
        builder.add_from_pipeline(registers, sequences, reports, evaluations)
        # When ready:
        X, Y = builder.to_numpy()
        dataset = builder.to_torch_dataset()
    """

    def __init__(
        self,
        feature_builder: Callable[[RegisterCandidate, SequenceCandidate], NDArray[np.float32]] | None = None,
        feature_dim: int = INPUT_DIM,
    ) -> None:
        self._features: list[NDArray[np.float32]] = []
        self._targets: list[NDArray[np.float32]] = []
        self._feature_builder = feature_builder or build_feature_vector
        self._feature_dim = feature_dim

    @property
    def size(self) -> int:
        return len(self._features)

    def add_sample(
        self,
        register: RegisterCandidate,
        sequence: SequenceCandidate,
        report: RobustnessReport,
        evaluation: EvaluationResult,
    ) -> None:
        self._features.append(self._feature_builder(register, sequence))
        self._targets.append(build_target_vector(report, evaluation))

    def add_from_pipeline(
        self,
        registers: list[RegisterCandidate],
        sequences: list[SequenceCandidate],
        reports: list[RobustnessReport],
        evaluations: list[EvaluationResult],
    ) -> int:
        """Match sequences to their register and report, add all valid samples.

        Returns the number of samples added.
        """
        reg_lookup = {r.id: r for r in registers}
        report_lookup = {r.sequence_candidate_id: r for r in reports}
        eval_lookup = {e.sequence_candidate_id: e for e in evaluations}

        added = 0
        for seq in sequences:
            reg = reg_lookup.get(seq.register_candidate_id)
            report = report_lookup.get(seq.id)
            evaluation = eval_lookup.get(seq.id)
            if reg and report and evaluation:
                self.add_sample(reg, seq, report, evaluation)
                added += 1
        return added

    def to_numpy(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not self._features:
            return (
                np.empty((0, self._feature_dim), dtype=np.float32),
                np.empty((0, OUTPUT_DIM), dtype=np.float32),
            )
        return np.stack(self._features), np.stack(self._targets)

    def to_torch_dataset(self) -> TensorDataset:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required: pip install 'cryoswarm-q[ml]'")
        X, Y = self.to_numpy()
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))

    def save(self, path: str) -> None:
        X, Y = self.to_numpy()
        np.savez(path, features=X, targets=Y)

    def load(self, path: str) -> None:
        data = np.load(path)
        for i in range(data["features"].shape[0]):
            self._features.append(data["features"][i])
            self._targets.append(data["targets"][i])
