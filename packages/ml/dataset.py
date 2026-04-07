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
from packages.core.logging import get_logger
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
_DEFAULT_PARAM_SPACE = PhysicsParameterSpace.default()
logger = get_logger(__name__)


def _max_detuning_domain(space: PhysicsParameterSpace) -> tuple[float, float]:
    detuning_mins = [float(pulse_space.detuning_start.min_val) for pulse_space in space.pulse.values()]
    detuning_maxs = [float(pulse_space.detuning_start.max_val) for pulse_space in space.pulse.values()]
    for pulse_space in space.pulse.values():
        if pulse_space.detuning_end is None:
            continue
        detuning_mins.append(float(pulse_space.detuning_end.min_val))
        detuning_maxs.append(float(pulse_space.detuning_end.max_val))
    return min(detuning_mins), max(detuning_maxs)


def _max_blockade_radius_um(space: PhysicsParameterSpace) -> float:
    min_amplitude = min(float(pulse_space.amplitude.min_val) for pulse_space in space.pulse.values())
    if min_amplitude <= 0.0:
        return float(space.default_blockade_radius_um)
    return float((space.c6_coefficient / min_amplitude) ** (1.0 / 6.0))


def _max_sweep_span(space: PhysicsParameterSpace) -> float:
    max_span = 0.0
    for pulse_space in space.pulse.values():
        if pulse_space.detuning_end is None:
            continue
        for detuning_start in (pulse_space.detuning_start.min_val, pulse_space.detuning_start.max_val):
            for detuning_end in (pulse_space.detuning_end.min_val, pulse_space.detuning_end.max_val):
                max_span = max(max_span, abs(float(detuning_end) - float(detuning_start)))
    return max_span


def _build_feature_normalization_constants(space: PhysicsParameterSpace) -> dict[str, float]:
    amplitude_scale = max(float(pulse_space.amplitude.max_val) for pulse_space in space.pulse.values())
    detuning_min, detuning_max = _max_detuning_domain(space)
    detuning_offset = abs(detuning_min)
    duration_ns_scale = max(float(pulse_space.duration_ns.max_val) for pulse_space in space.pulse.values())
    blockade_radius_scale = _max_blockade_radius_um(space)
    min_distance_scale = float(space.geometry.spacing_um.max_val)
    max_spacing = float(space.geometry.spacing_um.max_val)
    min_spacing = max(float(space.geometry.spacing_um.min_val), 1e-6)
    adiabaticity_scale = max(
        float(pulse_space.duration_ns.max_val) / 1000.0 * float(pulse_space.amplitude.max_val)
        for pulse_space in space.pulse.values()
    )
    interaction_floor = max(space.c6_coefficient / max(max_spacing**6, 1e-6), 1e-10)
    min_amplitude = max(
        min(float(pulse_space.amplitude.min_val) for pulse_space in space.pulse.values()),
        1e-10,
    )
    return {
        "amplitude_scale": amplitude_scale,
        "detuning_offset": detuning_offset,
        "detuning_range": float(detuning_max - detuning_min),
        "duration_ns_scale": duration_ns_scale,
        "layout_scale": float(max(len(LAYOUT_ENCODING) - 1, 1)),
        "family_scale": float(max(len(SequenceFamily) - 1, 1)),
        "omega_over_interaction_scale": amplitude_scale / interaction_floor,
        "detuning_over_omega_scale": max(abs(detuning_min), abs(detuning_max)) / min_amplitude,
        "adiabaticity_scale": adiabaticity_scale,
        "atom_count_scale": float(space.geometry.atom_count.max_val),
        "predicted_cost_log_scale": float(np.log1p(1.0)),
        "blockade_over_spacing_scale": blockade_radius_scale / min_spacing,
        "blockade_radius_scale": blockade_radius_scale,
        "min_distance_scale": min_distance_scale,
        "sweep_span_scale": _max_sweep_span(space),
    }


FEATURE_NORMALIZATION_DEFAULTS = _build_feature_normalization_constants(_DEFAULT_PARAM_SPACE)
AMPLITUDE_SCALE = FEATURE_NORMALIZATION_DEFAULTS["amplitude_scale"]
DETUNING_OFFSET = FEATURE_NORMALIZATION_DEFAULTS["detuning_offset"]
DETUNING_RANGE = FEATURE_NORMALIZATION_DEFAULTS["detuning_range"]
DURATION_NS_SCALE = FEATURE_NORMALIZATION_DEFAULTS["duration_ns_scale"]
LAYOUT_SCALE = FEATURE_NORMALIZATION_DEFAULTS["layout_scale"]
FAMILY_SCALE = FEATURE_NORMALIZATION_DEFAULTS["family_scale"]
OMEGA_OVER_INTERACTION_SCALE = FEATURE_NORMALIZATION_DEFAULTS["omega_over_interaction_scale"]
DETUNING_OVER_OMEGA_SCALE = FEATURE_NORMALIZATION_DEFAULTS["detuning_over_omega_scale"]
ADIABATICITY_SCALE = FEATURE_NORMALIZATION_DEFAULTS["adiabaticity_scale"]
ATOM_COUNT_SCALE = FEATURE_NORMALIZATION_DEFAULTS["atom_count_scale"]
PREDICTED_COST_LOG_SCALE = FEATURE_NORMALIZATION_DEFAULTS["predicted_cost_log_scale"]
BLOCKADE_OVER_SPACING_SCALE = FEATURE_NORMALIZATION_DEFAULTS["blockade_over_spacing_scale"]
BLOCKADE_RADIUS_SCALE = FEATURE_NORMALIZATION_DEFAULTS["blockade_radius_scale"]
MIN_DISTANCE_SCALE = FEATURE_NORMALIZATION_DEFAULTS["min_distance_scale"]
SWEEP_SPAN_SCALE = FEATURE_NORMALIZATION_DEFAULTS["sweep_span_scale"]


def feature_normalization_constants(
    param_space: PhysicsParameterSpace | None = None,
) -> dict[str, float]:
    space = param_space or _DEFAULT_PARAM_SPACE
    if space is _DEFAULT_PARAM_SPACE:
        return FEATURE_NORMALIZATION_DEFAULTS
    return _build_feature_normalization_constants(space)


def _encode_layout(name: str) -> int:
    return LAYOUT_ENCODING.get(name, len(LAYOUT_ENCODING))


def _encode_family(name: str) -> int:
    return FAMILY_ENCODING.get(name, len(FAMILY_ENCODING))


def build_feature_vector(
    register: RegisterCandidate,
    sequence: SequenceCandidate,
) -> NDArray[np.float32]:
    """Fixed-size input feature vector from a (register, sequence) pair."""
    spacing = float(register.metadata.get("spacing_um", _DEFAULT_PARAM_SPACE.geometry.spacing_um.default))
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
    normalizers = feature_normalization_constants(space)
    spacing = float(register.metadata.get("spacing_um", space.geometry.spacing_um.default))
    blockade_pairs_total = max(register.atom_count * (register.atom_count - 1) / 2, 1)
    blockade_fraction = float(register.blockade_pair_count / blockade_pairs_total)

    interaction_energy = space.c6_coefficient / (spacing**6) if spacing > 0 else 0.0
    omega_over_interaction = sequence.amplitude / max(interaction_energy, 1e-10) if interaction_energy else 0.0
    omega_over_interaction_norm = min(omega_over_interaction, normalizers["omega_over_interaction_scale"]) / max(
        normalizers["omega_over_interaction_scale"],
        1e-10,
    )

    detuning_over_omega = abs(sequence.detuning) / max(sequence.amplitude, 0.01)
    detuning_over_omega_norm = min(detuning_over_omega, normalizers["detuning_over_omega_scale"]) / max(
        normalizers["detuning_over_omega_scale"],
        1e-10,
    )

    duration_us = sequence.duration_ns / 1000.0
    adiabaticity = duration_us * sequence.amplitude
    adiabaticity_norm = min(adiabaticity, normalizers["adiabaticity_scale"]) / max(
        normalizers["adiabaticity_scale"],
        1e-10,
    )

    hilbert_dim_log = register.atom_count / max(normalizers["atom_count_scale"], 1.0)
    feasibility = float(register.feasibility_score)
    cost_log = min(np.log1p(sequence.predicted_cost), normalizers["predicted_cost_log_scale"]) / max(
        normalizers["predicted_cost_log_scale"],
        1e-10,
    )
    blockade_over_spacing = register.blockade_radius_um / max(spacing, 0.1)
    blockade_over_spacing_norm = min(blockade_over_spacing, normalizers["blockade_over_spacing_scale"]) / max(
        normalizers["blockade_over_spacing_scale"],
        1e-10,
    )

    detuning_end = float(sequence.metadata.get("detuning_end", sequence.detuning))
    sweep_span_norm = min(abs(detuning_end - sequence.detuning), normalizers["sweep_span_scale"]) / max(
        normalizers["sweep_span_scale"],
        1e-10,
    )

    return np.array(
        [
            register.atom_count / max(normalizers["atom_count_scale"], 1.0),
            spacing / max(float(space.geometry.spacing_um.max_val), 1.0),
            sequence.amplitude / max(normalizers["amplitude_scale"], 1.0),
            (sequence.detuning + normalizers["detuning_offset"]) / max(normalizers["detuning_range"], 1.0),
            sequence.duration_ns / max(normalizers["duration_ns_scale"], 1.0),
            _encode_layout(register.layout_type) / max(normalizers["layout_scale"], 1.0),
            _encode_family(sequence.sequence_family.value) / max(normalizers["family_scale"], 1.0),
            blockade_fraction,
            omega_over_interaction_norm,
            detuning_over_omega_norm,
            adiabaticity_norm,
            hilbert_dim_log,
            feasibility,
            cost_log,
            blockade_over_spacing_norm,
            register.blockade_radius_um / max(normalizers["blockade_radius_scale"], 1.0),
            register.min_distance_um / max(normalizers["min_distance_scale"], 1.0),
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
        dropped = 0
        for seq in sequences:
            reg = reg_lookup.get(seq.register_candidate_id)
            report = report_lookup.get(seq.id)
            evaluation = eval_lookup.get(seq.id)
            if reg and report and evaluation:
                self.add_sample(reg, seq, report, evaluation)
                added += 1
            else:
                dropped += 1
        logger.info(
            "Pipeline ingestion: %d accepted, %d dropped (missing pairs)",
            added,
            dropped,
        )
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
