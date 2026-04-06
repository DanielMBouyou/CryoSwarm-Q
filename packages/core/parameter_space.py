"""Physics parameter space for neutral-atom experiment design.

Centralizes all tunable physical parameters across CryoSwarm-Q agents.
Replaces hardcoded constants with configurable, samplable ranges.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from packages.core.enums import NoiseLevel, SequenceFamily
from packages.core.models import NoiseScenario

try:
    from scipy.stats.qmc import LatinHypercube

    SCIPY_QMC_AVAILABLE = True
except ImportError:  # pragma: no cover - optional fallback
    LatinHypercube = None  # type: ignore[assignment]
    SCIPY_QMC_AVAILABLE = False


def _quantize(value: float, step: float | None) -> float:
    if step is None or step <= 0:
        return float(value)
    return round(round(float(value) / step) * step, 10)


def _latin_hypercube_unit(
    n_samples: int,
    n_dims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_samples <= 0 or n_dims <= 0:
        return np.empty((0, n_dims), dtype=np.float32)

    if SCIPY_QMC_AVAILABLE:
        sampler = LatinHypercube(d=n_dims, seed=rng)
        return sampler.random(n=n_samples).astype(np.float32)

    result = np.zeros((n_samples, n_dims), dtype=np.float32)
    intervals = np.linspace(0.0, 1.0, n_samples + 1, dtype=np.float32)
    for dim in range(n_dims):
        points = rng.uniform(intervals[:-1], intervals[1:])
        rng.shuffle(points)
        result[:, dim] = points
    return result


@dataclass
class ParameterRange:
    """A single tunable physical parameter with bounds and metadata."""

    name: str
    min_val: float
    max_val: float
    default: float
    unit: str
    description: str
    log_scale: bool = False
    quantize: float | None = None

    def __post_init__(self) -> None:
        if self.max_val < self.min_val:
            raise ValueError(f"{self.name}: max_val must be >= min_val.")
        if not self.min_val <= self.default <= self.max_val:
            raise ValueError(f"{self.name}: default must lie inside [min_val, max_val].")
        if self.log_scale and self.min_val <= 0:
            raise ValueError(f"{self.name}: log-scale sampling requires min_val > 0.")

    def sample_uniform(self, rng: np.random.Generator) -> float:
        return self.clip(rng.uniform(self.min_val, self.max_val))

    def sample_log_uniform(self, rng: np.random.Generator) -> float:
        log_min = math.log(self.min_val)
        log_max = math.log(self.max_val)
        return self.clip(math.exp(rng.uniform(log_min, log_max)))

    def sample(self, rng: np.random.Generator) -> float:
        if self.log_scale:
            return self.sample_log_uniform(rng)
        return self.sample_uniform(rng)

    def clip(self, value: float) -> float:
        clipped = min(max(float(value), self.min_val), self.max_val)
        quantized = _quantize(clipped, self.quantize)
        return min(max(float(quantized), self.min_val), self.max_val)

    def normalize(self, value: float) -> float:
        if math.isclose(self.max_val, self.min_val):
            return 0.0
        return float((self.clip(value) - self.min_val) / (self.max_val - self.min_val))

    def denormalize(self, normalized: float) -> float:
        value = self.min_val + float(np.clip(normalized, 0.0, 1.0)) * (self.max_val - self.min_val)
        return self.clip(value)

    def grid(self, n_points: int) -> list[float]:
        if n_points <= 1:
            return [self.clip(self.default)]
        values = np.linspace(self.min_val, self.max_val, n_points)
        return [self.clip(float(value)) for value in values]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "default": self.default,
            "unit": self.unit,
            "description": self.description,
            "log_scale": self.log_scale,
            "quantize": self.quantize,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParameterRange":
        return cls(
            name=str(payload["name"]),
            min_val=float(payload["min_val"]),
            max_val=float(payload["max_val"]),
            default=float(payload["default"]),
            unit=str(payload["unit"]),
            description=str(payload["description"]),
            log_scale=bool(payload.get("log_scale", False)),
            quantize=float(payload["quantize"]) if payload.get("quantize") is not None else None,
        )


@dataclass
class PulseParameterSpace:
    """Parameter space for a single pulse-sequence family."""

    family: SequenceFamily
    amplitude: ParameterRange
    detuning_start: ParameterRange
    detuning_end: ParameterRange | None
    duration_ns: ParameterRange
    amplitude_start: ParameterRange | None
    phase: ParameterRange

    def applicable_ranges(self) -> dict[str, ParameterRange]:
        ranges = {
            "amplitude": self.amplitude,
            "detuning": self.detuning_start,
            "duration_ns": self.duration_ns,
            "phase": self.phase,
        }
        if self.detuning_end is not None:
            ranges["detuning_end"] = self.detuning_end
        if self.amplitude_start is not None:
            ranges["amplitude_start"] = self.amplitude_start
        return ranges

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "amplitude": self.amplitude.to_dict(),
            "detuning_start": self.detuning_start.to_dict(),
            "detuning_end": self.detuning_end.to_dict() if self.detuning_end is not None else None,
            "duration_ns": self.duration_ns.to_dict(),
            "amplitude_start": self.amplitude_start.to_dict() if self.amplitude_start is not None else None,
            "phase": self.phase.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PulseParameterSpace":
        return cls(
            family=SequenceFamily(str(payload["family"])),
            amplitude=ParameterRange.from_dict(payload["amplitude"]),
            detuning_start=ParameterRange.from_dict(payload["detuning_start"]),
            detuning_end=ParameterRange.from_dict(payload["detuning_end"]) if payload.get("detuning_end") else None,
            duration_ns=ParameterRange.from_dict(payload["duration_ns"]),
            amplitude_start=ParameterRange.from_dict(payload["amplitude_start"]) if payload.get("amplitude_start") else None,
            phase=ParameterRange.from_dict(payload["phase"]),
        )


@dataclass
class GeometryParameterSpace:
    """Parameter space for register geometry generation."""

    spacing_um: ParameterRange
    min_spacing_um: ParameterRange
    atom_count: ParameterRange
    feasibility_base: ParameterRange
    feasibility_spacing_weight: ParameterRange
    feasibility_blockade_weight: ParameterRange
    feasibility_atom_penalty: ParameterRange

    def to_dict(self) -> dict[str, Any]:
        return {
            "spacing_um": self.spacing_um.to_dict(),
            "min_spacing_um": self.min_spacing_um.to_dict(),
            "atom_count": self.atom_count.to_dict(),
            "feasibility_base": self.feasibility_base.to_dict(),
            "feasibility_spacing_weight": self.feasibility_spacing_weight.to_dict(),
            "feasibility_blockade_weight": self.feasibility_blockade_weight.to_dict(),
            "feasibility_atom_penalty": self.feasibility_atom_penalty.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GeometryParameterSpace":
        return cls(
            spacing_um=ParameterRange.from_dict(payload["spacing_um"]),
            min_spacing_um=ParameterRange.from_dict(payload["min_spacing_um"]),
            atom_count=ParameterRange.from_dict(payload["atom_count"]),
            feasibility_base=ParameterRange.from_dict(payload["feasibility_base"]),
            feasibility_spacing_weight=ParameterRange.from_dict(payload["feasibility_spacing_weight"]),
            feasibility_blockade_weight=ParameterRange.from_dict(payload["feasibility_blockade_weight"]),
            feasibility_atom_penalty=ParameterRange.from_dict(payload["feasibility_atom_penalty"]),
        )


@dataclass
class NoiseParameterSpace:
    """Parameter space for noise model configuration."""

    amplitude_jitter: ParameterRange
    detuning_jitter: ParameterRange
    dephasing_rate: ParameterRange
    atom_loss_rate: ParameterRange
    temperature_uk: ParameterRange
    state_prep_error: ParameterRange
    false_positive_rate: ParameterRange
    false_negative_rate: ParameterRange

    def to_dict(self) -> dict[str, Any]:
        return {
            "amplitude_jitter": self.amplitude_jitter.to_dict(),
            "detuning_jitter": self.detuning_jitter.to_dict(),
            "dephasing_rate": self.dephasing_rate.to_dict(),
            "atom_loss_rate": self.atom_loss_rate.to_dict(),
            "temperature_uk": self.temperature_uk.to_dict(),
            "state_prep_error": self.state_prep_error.to_dict(),
            "false_positive_rate": self.false_positive_rate.to_dict(),
            "false_negative_rate": self.false_negative_rate.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NoiseParameterSpace":
        return cls(
            amplitude_jitter=ParameterRange.from_dict(payload["amplitude_jitter"]),
            detuning_jitter=ParameterRange.from_dict(payload["detuning_jitter"]),
            dephasing_rate=ParameterRange.from_dict(payload["dephasing_rate"]),
            atom_loss_rate=ParameterRange.from_dict(payload["atom_loss_rate"]),
            temperature_uk=ParameterRange.from_dict(payload["temperature_uk"]),
            state_prep_error=ParameterRange.from_dict(payload["state_prep_error"]),
            false_positive_rate=ParameterRange.from_dict(payload["false_positive_rate"]),
            false_negative_rate=ParameterRange.from_dict(payload["false_negative_rate"]),
        )


@dataclass
class ScoringParameterSpace:
    """Parameter space for scoring and robustness weights."""

    nominal_weight: ParameterRange
    average_weight: ParameterRange
    worst_case_weight: ParameterRange
    stability_weight: ParameterRange
    density_score_weight: ParameterRange
    blockade_score_weight: ParameterRange
    stability_std_threshold: ParameterRange

    def to_dict(self) -> dict[str, Any]:
        return {
            "nominal_weight": self.nominal_weight.to_dict(),
            "average_weight": self.average_weight.to_dict(),
            "worst_case_weight": self.worst_case_weight.to_dict(),
            "stability_weight": self.stability_weight.to_dict(),
            "density_score_weight": self.density_score_weight.to_dict(),
            "blockade_score_weight": self.blockade_score_weight.to_dict(),
            "stability_std_threshold": self.stability_std_threshold.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScoringParameterSpace":
        return cls(
            nominal_weight=ParameterRange.from_dict(payload["nominal_weight"]),
            average_weight=ParameterRange.from_dict(payload["average_weight"]),
            worst_case_weight=ParameterRange.from_dict(payload["worst_case_weight"]),
            stability_weight=ParameterRange.from_dict(payload["stability_weight"]),
            density_score_weight=ParameterRange.from_dict(payload["density_score_weight"]),
            blockade_score_weight=ParameterRange.from_dict(payload["blockade_score_weight"]),
            stability_std_threshold=ParameterRange.from_dict(payload["stability_std_threshold"]),
        )


@dataclass
class PhysicsParameterSpace:
    """Master parameter space aggregating all sub-spaces."""

    pulse: dict[SequenceFamily, PulseParameterSpace]
    geometry: GeometryParameterSpace
    noise: NoiseParameterSpace
    scoring: ScoringParameterSpace
    duration_offset_per_atom: ParameterRange
    cost_base_exponent: ParameterRange
    cost_normalization: ParameterRange

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def default(cls) -> "PhysicsParameterSpace":
        phase = ParameterRange("phase", 0.0, 2.0 * math.pi, 0.0, "rad", "Pulse phase.")
        pulse = {
            SequenceFamily.ADIABATIC_SWEEP: PulseParameterSpace(
                family=SequenceFamily.ADIABATIC_SWEEP,
                amplitude=ParameterRange("adiabatic_amplitude", 1.0, 15.0, 5.0, "rad/us", "Adiabatic sweep Rabi frequency."),
                detuning_start=ParameterRange("adiabatic_detuning_start", -40.0, -5.0, -20.0, "rad/us", "Initial detuning."),
                detuning_end=ParameterRange("adiabatic_detuning_end", 5.0, 25.0, 10.0, "rad/us", "Final detuning."),
                duration_ns=ParameterRange("adiabatic_duration_ns", 1500.0, 6000.0, 3000.0, "ns", "Sweep duration.", quantize=4.0),
                amplitude_start=ParameterRange("adiabatic_amplitude_start", 0.1, 2.0, 0.5, "rad/us", "Ramp start amplitude."),
                phase=phase,
            ),
            SequenceFamily.DETUNING_SCAN: PulseParameterSpace(
                family=SequenceFamily.DETUNING_SCAN,
                amplitude=ParameterRange("detuning_scan_amplitude", 2.0, 12.0, 4.0, "rad/us", "Scan Rabi frequency."),
                detuning_start=ParameterRange("detuning_scan_detuning_start", -30.0, -5.0, -15.0, "rad/us", "Initial detuning."),
                detuning_end=ParameterRange("detuning_scan_detuning_end", 0.0, 20.0, 5.0, "rad/us", "Final detuning."),
                duration_ns=ParameterRange("detuning_scan_duration_ns", 1000.0, 5000.0, 2000.0, "ns", "Scan duration.", quantize=4.0),
                amplitude_start=None,
                phase=phase,
            ),
            SequenceFamily.GLOBAL_RAMP: PulseParameterSpace(
                family=SequenceFamily.GLOBAL_RAMP,
                amplitude=ParameterRange("global_ramp_amplitude", 3.0, 15.0, 6.0, "rad/us", "Ramp target amplitude."),
                detuning_start=ParameterRange("global_ramp_detuning_start", -30.0, -5.0, -12.0, "rad/us", "Drive detuning."),
                detuning_end=None,
                duration_ns=ParameterRange("global_ramp_duration_ns", 1000.0, 5000.0, 2000.0, "ns", "Ramp duration.", quantize=4.0),
                amplitude_start=ParameterRange("global_ramp_amplitude_start", 0.1, 3.0, 0.5, "rad/us", "Ramp start amplitude."),
                phase=phase,
            ),
            SequenceFamily.CONSTANT_DRIVE: PulseParameterSpace(
                family=SequenceFamily.CONSTANT_DRIVE,
                amplitude=ParameterRange("constant_drive_amplitude", 1.0, 10.0, 5.0, "rad/us", "Constant drive amplitude."),
                detuning_start=ParameterRange("constant_drive_detuning", -10.0, 0.0, 0.0, "rad/us", "Constant detuning."),
                detuning_end=None,
                duration_ns=ParameterRange("constant_drive_duration_ns", 500.0, 3000.0, 1200.0, "ns", "Drive duration.", quantize=4.0),
                amplitude_start=None,
                phase=phase,
            ),
            SequenceFamily.BLACKMAN_SWEEP: PulseParameterSpace(
                family=SequenceFamily.BLACKMAN_SWEEP,
                amplitude=ParameterRange("blackman_sweep_amplitude", 3.0, 15.0, 7.0, "rad/us", "Blackman sweep amplitude."),
                detuning_start=ParameterRange("blackman_sweep_detuning_start", -40.0, -10.0, -20.0, "rad/us", "Initial detuning."),
                detuning_end=ParameterRange("blackman_sweep_detuning_end", 5.0, 25.0, 10.0, "rad/us", "Final detuning."),
                duration_ns=ParameterRange("blackman_sweep_duration_ns", 1500.0, 6000.0, 3000.0, "ns", "Sweep duration.", quantize=4.0),
                amplitude_start=None,
                phase=phase,
            ),
        }
        return cls(
            pulse=pulse,
            geometry=GeometryParameterSpace(
                spacing_um=ParameterRange("spacing_um", 4.0, 15.0, 7.0, "um", "Inter-atom spacing."),
                min_spacing_um=ParameterRange("min_spacing_um", 3.5, 6.0, 5.0, "um", "Minimum safe distance."),
                atom_count=ParameterRange("atom_count", 2.0, 25.0, 6.0, "atoms", "Register atom count.", quantize=1.0),
                feasibility_base=ParameterRange("feasibility_base", 0.3, 0.7, 0.55, "-", "Base feasibility score."),
                feasibility_spacing_weight=ParameterRange("feasibility_spacing_weight", 0.05, 0.40, 0.20, "-", "Spacing contribution."),
                feasibility_blockade_weight=ParameterRange("feasibility_blockade_weight", 0.05, 0.40, 0.20, "-", "Blockade contribution."),
                feasibility_atom_penalty=ParameterRange("feasibility_atom_penalty", 0.01, 0.10, 0.04, "per_atom", "Large-register penalty."),
            ),
            noise=NoiseParameterSpace(
                amplitude_jitter=ParameterRange("amplitude_jitter", 0.01, 0.15, 0.06, "fractional", "Fractional amplitude jitter."),
                detuning_jitter=ParameterRange("detuning_jitter", 0.01, 0.12, 0.05, "fractional", "Fractional detuning jitter."),
                dephasing_rate=ParameterRange("dephasing_rate", 0.01, 0.15, 0.07, "us^-1", "Dephasing rate."),
                atom_loss_rate=ParameterRange("atom_loss_rate", 0.005, 0.08, 0.03, "us^-1", "Atom loss rate."),
                temperature_uk=ParameterRange("temperature_uk", 10.0, 100.0, 50.0, "uK", "Atom temperature."),
                state_prep_error=ParameterRange("state_prep_error", 0.001, 0.02, 0.005, "-", "State preparation error."),
                false_positive_rate=ParameterRange("false_positive_rate", 0.003, 0.05, 0.01, "-", "Detection false positive rate."),
                false_negative_rate=ParameterRange("false_negative_rate", 0.01, 0.10, 0.05, "-", "Detection false negative rate."),
            ),
            scoring=ScoringParameterSpace(
                nominal_weight=ParameterRange("nominal_weight", 0.10, 0.40, 0.25, "-", "Nominal robustness weight."),
                average_weight=ParameterRange("average_weight", 0.15, 0.50, 0.35, "-", "Average perturbed robustness weight."),
                worst_case_weight=ParameterRange("worst_case_weight", 0.10, 0.50, 0.30, "-", "Worst-case robustness weight."),
                stability_weight=ParameterRange("stability_weight", 0.05, 0.20, 0.10, "-", "Stability weight."),
                density_score_weight=ParameterRange("density_score_weight", 0.4, 0.9, 0.7, "-", "Density observable weight."),
                blockade_score_weight=ParameterRange("blockade_score_weight", 0.1, 0.6, 0.3, "-", "Blockade observable weight."),
                stability_std_threshold=ParameterRange("stability_std_threshold", 0.05, 0.50, 0.20, "-", "Std threshold for stability bonus."),
            ),
            duration_offset_per_atom=ParameterRange("duration_offset_per_atom", 50.0, 300.0, 150.0, "ns/atom", "Sequence duration offset per atom."),
            cost_base_exponent=ParameterRange("cost_base_exponent", 1.6, 2.0, 2.0, "-", "Exponential cost base per atom."),
            cost_normalization=ParameterRange("cost_normalization", 100000.0, 1000000.0, 500000.0, "-", "Cost normalization factor."),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "PhysicsParameterSpace":
        content = Path(path).read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore

            payload = yaml.safe_load(content)
        except ImportError:
            payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("Serialized parameter space must decode to a mapping.")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhysicsParameterSpace":
        pulse = {
            SequenceFamily(str(family_key)): PulseParameterSpace.from_dict(family_payload)
            for family_key, family_payload in payload["pulse"].items()
        }
        return cls(
            pulse=pulse,
            geometry=GeometryParameterSpace.from_dict(payload["geometry"]),
            noise=NoiseParameterSpace.from_dict(payload["noise"]),
            scoring=ScoringParameterSpace.from_dict(payload["scoring"]),
            duration_offset_per_atom=ParameterRange.from_dict(payload["duration_offset_per_atom"]),
            cost_base_exponent=ParameterRange.from_dict(payload["cost_base_exponent"]),
            cost_normalization=ParameterRange.from_dict(payload["cost_normalization"]),
        )

    def to_yaml(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pulse": {family.value: pulse_space.to_dict() for family, pulse_space in self.pulse.items()},
            "geometry": self.geometry.to_dict(),
            "noise": self.noise.to_dict(),
            "scoring": self.scoring.to_dict(),
            "duration_offset_per_atom": self.duration_offset_per_atom.to_dict(),
            "cost_base_exponent": self.cost_base_exponent.to_dict(),
            "cost_normalization": self.cost_normalization.to_dict(),
        }

    def validate(self) -> None:
        missing = set(SequenceFamily).difference(self.pulse)
        if missing:
            raise ValueError(f"Missing pulse parameter spaces for: {sorted(item.value for item in missing)}")

        scoring_sum = (
            self.scoring.nominal_weight.default
            + self.scoring.average_weight.default
            + self.scoring.worst_case_weight.default
            + self.scoring.stability_weight.default
        )
        if not math.isclose(scoring_sum, 1.0, abs_tol=1e-6):
            raise ValueError("Default scoring robustness weights must sum to 1.0.")

        observable_sum = self.scoring.density_score_weight.default + self.scoring.blockade_score_weight.default
        if not math.isclose(observable_sum, 1.0, abs_tol=1e-6):
            raise ValueError("Observable score weights must sum to 1.0.")

        for family, pulse_space in self.pulse.items():
            if pulse_space.detuning_end is not None and pulse_space.detuning_end.default <= pulse_space.detuning_start.default:
                raise ValueError(f"{family.value}: detuning_end default must exceed detuning_start default.")
            if pulse_space.amplitude_start is not None and pulse_space.amplitude_start.default > pulse_space.amplitude.default:
                raise ValueError(f"{family.value}: amplitude_start default must not exceed amplitude default.")

        if self.geometry.min_spacing_um.default > self.geometry.spacing_um.max_val:
            raise ValueError("Default minimum spacing cannot exceed maximum nominal spacing.")

    def duration_offset(self, atom_count: int) -> int:
        return int(round(self.duration_offset_per_atom.default * atom_count))

    def cost_for(self, atom_count: int, duration_ns: int) -> float:
        return round(
            min(
                1.0,
                (self.cost_base_exponent.default**atom_count) * duration_ns / self.cost_normalization.default,
            ),
            4,
        )

    def rl_action_ranges(self) -> dict[str, tuple[float, float]]:
        amplitude_min = min(space.amplitude.min_val for space in self.pulse.values())
        amplitude_max = min(15.0, max(space.amplitude.max_val for space in self.pulse.values()))
        detuning_min = max(-30.0, min(space.detuning_start.min_val for space in self.pulse.values()))
        detuning_max = min(
            15.0,
            max(
                [space.detuning_start.max_val for space in self.pulse.values()]
                + [space.detuning_end.max_val for space in self.pulse.values() if space.detuning_end is not None]
            ),
        )
        duration_min = min(space.duration_ns.min_val for space in self.pulse.values())
        duration_max = min(5500.0, max(space.duration_ns.max_val for space in self.pulse.values()))
        return {
            "amplitude": (float(amplitude_min), float(amplitude_max)),
            "detuning": (float(detuning_min), float(detuning_max)),
            "duration_ns": (float(duration_min), float(duration_max)),
        }

    def noise_profile(self, level: NoiseLevel) -> NoiseScenario:
        payload = {
            NoiseLevel.LOW: {
                "amplitude_jitter": 0.03,
                "detuning_jitter": 0.02,
                "dephasing_rate": 0.03,
                "atom_loss_rate": 0.01,
                "temperature_uk": 30.0,
                "state_prep_error": 0.003,
                "false_positive_rate": 0.008,
                "false_negative_rate": 0.02,
                "metadata": {"spatial_inhomogeneity": 0.02},
            },
            NoiseLevel.MEDIUM: {
                "amplitude_jitter": self.noise.amplitude_jitter.default,
                "detuning_jitter": self.noise.detuning_jitter.default,
                "dephasing_rate": self.noise.dephasing_rate.default,
                "atom_loss_rate": self.noise.atom_loss_rate.default,
                "temperature_uk": self.noise.temperature_uk.default,
                "state_prep_error": self.noise.state_prep_error.default,
                "false_positive_rate": self.noise.false_positive_rate.default,
                "false_negative_rate": self.noise.false_negative_rate.default,
                "metadata": {"spatial_inhomogeneity": 0.05},
            },
            NoiseLevel.STRESSED: {
                "amplitude_jitter": 0.10,
                "detuning_jitter": 0.08,
                "dephasing_rate": 0.11,
                "atom_loss_rate": 0.05,
                "temperature_uk": 75.0,
                "state_prep_error": 0.01,
                "false_positive_rate": 0.02,
                "false_negative_rate": 0.08,
                "metadata": {"spatial_inhomogeneity": 0.08},
            },
        }[level]
        return NoiseScenario(label=level, **payload)

    def default_noise_scenarios(self) -> list[NoiseScenario]:
        return [
            self.noise_profile(NoiseLevel.LOW),
            self.noise_profile(NoiseLevel.MEDIUM),
            self.noise_profile(NoiseLevel.STRESSED),
        ]

    def sample_noise_scenario(self, rng: np.random.Generator) -> NoiseScenario:
        sampled = {
            "amplitude_jitter": self.noise.amplitude_jitter.sample(rng),
            "detuning_jitter": self.noise.detuning_jitter.sample(rng),
            "dephasing_rate": self.noise.dephasing_rate.sample(rng),
            "atom_loss_rate": self.noise.atom_loss_rate.sample(rng),
            "temperature_uk": self.noise.temperature_uk.sample(rng),
            "state_prep_error": self.noise.state_prep_error.sample(rng),
            "false_positive_rate": self.noise.false_positive_rate.sample(rng),
            "false_negative_rate": self.noise.false_negative_rate.sample(rng),
        }
        severity = np.mean(
            [
                self.noise.amplitude_jitter.normalize(sampled["amplitude_jitter"]),
                self.noise.detuning_jitter.normalize(sampled["detuning_jitter"]),
                self.noise.dephasing_rate.normalize(sampled["dephasing_rate"]),
                self.noise.atom_loss_rate.normalize(sampled["atom_loss_rate"]),
            ]
        )
        label = NoiseLevel.LOW if severity < 0.33 else NoiseLevel.MEDIUM if severity < 0.66 else NoiseLevel.STRESSED
        return NoiseScenario(label=label, metadata={"sampled": True}, **sampled)

    def sample_pulse_config(
        self,
        family: SequenceFamily,
        atom_count: int,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        pulse_space = self.pulse[family]
        config: dict[str, float] = {
            "family": family.value,  # type: ignore[assignment]
            "amplitude": pulse_space.amplitude.sample(rng),
            "detuning": pulse_space.detuning_start.sample(rng),
            "duration_ns": pulse_space.duration_ns.clip(
                pulse_space.duration_ns.sample(rng) + self.duration_offset(atom_count)
            ),
            "phase": pulse_space.phase.sample(rng),
        }
        if pulse_space.detuning_end is not None:
            detuning_end = pulse_space.detuning_end.sample(rng)
            if detuning_end <= config["detuning"]:
                detuning_end = pulse_space.detuning_end.clip(config["detuning"] + abs(detuning_end) + 1.0)
            config["detuning_end"] = detuning_end
        if pulse_space.amplitude_start is not None:
            config["amplitude_start"] = min(pulse_space.amplitude_start.sample(rng), config["amplitude"])
        return config

    def grid_search_configs(
        self,
        family: SequenceFamily,
        n_amplitude: int = 10,
        n_detuning: int = 10,
        n_duration: int = 5,
    ) -> list[dict[str, float]]:
        pulse_space = self.pulse[family]
        configs: list[dict[str, float]] = []
        for amplitude in pulse_space.amplitude.grid(n_amplitude):
            for detuning in pulse_space.detuning_start.grid(n_detuning):
                for duration in pulse_space.duration_ns.grid(n_duration):
                    config: dict[str, float] = {
                        "family": family.value,  # type: ignore[assignment]
                        "amplitude": amplitude,
                        "detuning": detuning,
                        "duration_ns": duration,
                        "phase": pulse_space.phase.default,
                    }
                    if pulse_space.detuning_end is not None:
                        config["detuning_end"] = pulse_space.detuning_end.default
                    if pulse_space.amplitude_start is not None:
                        config["amplitude_start"] = min(pulse_space.amplitude_start.default, amplitude)
                    configs.append(config)
        return configs

    def latin_hypercube_sample(
        self,
        family: SequenceFamily,
        n_samples: int,
        atom_count: int,
        rng: np.random.Generator,
    ) -> list[dict[str, float]]:
        pulse_space = self.pulse[family]
        applicable = pulse_space.applicable_ranges()
        names = list(applicable.keys())
        lhs = _latin_hypercube_unit(n_samples, len(names), rng)
        configs: list[dict[str, float]] = []
        for row in lhs:
            config: dict[str, float] = {"family": family.value}  # type: ignore[assignment]
            for idx, name in enumerate(names):
                value = applicable[name].denormalize(float(row[idx]))
                if name == "duration_ns":
                    value = applicable[name].clip(value + self.duration_offset(atom_count))
                config[name] = value
            if "amplitude_start" in config:
                config["amplitude_start"] = min(config["amplitude_start"], config["amplitude"])
            if "detuning_end" in config and config["detuning_end"] <= config["detuning"]:
                config["detuning_end"] = pulse_space.detuning_end.clip(config["detuning"] + 1.0)  # type: ignore[union-attr]
            configs.append(config)
        return configs
