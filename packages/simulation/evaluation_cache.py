from __future__ import annotations

"""Content-addressable cache for simulation and evaluator results."""

from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
import json
from threading import RLock
from typing import Any

from packages.core.models import ExperimentSpec, NoiseScenario, RegisterCandidate, SequenceCandidate
from packages.core.parameter_space import PhysicsParameterSpace


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _semantic_spec_signature(spec: ExperimentSpec) -> dict[str, Any]:
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    goal_constraints = metadata.get("goal_constraints", {})
    return {
        "objective_class": spec.objective_class,
        "target_observable": spec.target_observable,
        "target_density": spec.target_density,
        "perturbation_budget": spec.perturbation_budget,
        "scoring_weights": spec.scoring_weights.model_dump(mode="json"),
        "latency_budget": spec.latency_budget,
        "goal_constraints": goal_constraints if isinstance(goal_constraints, dict) else {},
    }


def _semantic_register_signature(register_candidate: RegisterCandidate) -> dict[str, Any]:
    return {
        "layout_type": register_candidate.layout_type,
        "atom_count": register_candidate.atom_count,
        "coordinates": register_candidate.coordinates,
        "min_distance_um": register_candidate.min_distance_um,
        "blockade_radius_um": register_candidate.blockade_radius_um,
        "blockade_pair_count": register_candidate.blockade_pair_count,
        "van_der_waals_matrix": register_candidate.van_der_waals_matrix,
        "device_constraints": register_candidate.device_constraints,
        "metadata": register_candidate.metadata,
    }


def _semantic_sequence_signature(sequence_candidate: SequenceCandidate) -> dict[str, Any]:
    return {
        "sequence_family": sequence_candidate.sequence_family.value,
        "channel_id": sequence_candidate.channel_id,
        "duration_ns": sequence_candidate.duration_ns,
        "amplitude": sequence_candidate.amplitude,
        "detuning": sequence_candidate.detuning,
        "phase": sequence_candidate.phase,
        "waveform_kind": sequence_candidate.waveform_kind,
        "serialized_pulser_sequence": sequence_candidate.serialized_pulser_sequence,
        "metadata": sequence_candidate.metadata,
    }


def _semantic_noise_signature(noise_scenario: NoiseScenario | None) -> dict[str, Any] | None:
    if noise_scenario is None:
        return None
    return {
        "label": noise_scenario.label.value,
        "amplitude_jitter": noise_scenario.amplitude_jitter,
        "detuning_jitter": noise_scenario.detuning_jitter,
        "dephasing_rate": noise_scenario.dephasing_rate,
        "atom_loss_rate": noise_scenario.atom_loss_rate,
        "temperature_uk": noise_scenario.temperature_uk,
        "state_prep_error": noise_scenario.state_prep_error,
        "false_positive_rate": noise_scenario.false_positive_rate,
        "false_negative_rate": noise_scenario.false_negative_rate,
        "metadata": noise_scenario.metadata,
    }


def build_simulation_cache_key(
    spec: ExperimentSpec,
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
    noise_scenario: NoiseScenario | None,
    param_space: PhysicsParameterSpace,
    *,
    emulator_available: bool,
    pulser_available: bool,
    scipy_sparse_available: bool,
) -> str:
    payload = {
        "spec": _semantic_spec_signature(spec),
        "register": _semantic_register_signature(register_candidate),
        "sequence": _semantic_sequence_signature(sequence_candidate),
        "noise_scenario": _semantic_noise_signature(noise_scenario),
        "param_space": param_space.to_dict(),
        "backend_flags": {
            "emulator_available": emulator_available,
            "pulser_available": pulser_available,
            "scipy_sparse_available": scipy_sparse_available,
        },
    }
    return sha256(_canonical_json(payload).encode("ascii")).hexdigest()


class ContentAddressableEvaluationCache:
    """Small in-memory LRU cache keyed by deterministic content hashes."""

    def __init__(self, max_entries: int = 2048) -> None:
        self.max_entries = max_entries
        self._lock = RLock()
        self._entries: OrderedDict[str, tuple[float, dict[str, Any], dict[str, Any]]] = OrderedDict()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def get(self, key: str) -> tuple[float, dict[str, Any], dict[str, Any]] | None:
        with self._lock:
            payload = self._entries.pop(key, None)
            if payload is None:
                return None
            self._entries[key] = payload
            return deepcopy(payload)

    def set(self, key: str, value: tuple[float, dict[str, Any], dict[str, Any]]) -> None:
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = deepcopy(value)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_EVALUATION_CACHE = ContentAddressableEvaluationCache()


def get_evaluation_cache() -> ContentAddressableEvaluationCache:
    return _EVALUATION_CACHE


def clear_evaluation_cache() -> None:
    _EVALUATION_CACHE.clear()
