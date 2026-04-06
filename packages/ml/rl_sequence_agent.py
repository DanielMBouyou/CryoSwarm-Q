"""Phase 2 - RL-driven sequence agent."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from packages.agents.base import BaseAgent
from packages.agents.protocols import SequenceProtocol
from packages.agents.sequence_agent import SequenceAgent
from packages.core.enums import AgentName
from packages.core.logging import get_logger
from packages.core.models import ExperimentSpec, MemoryRecord, RegisterCandidate, SequenceCandidate
from packages.core.parameter_space import PhysicsParameterSpace
from packages.ml.rl_env import rescale_action
from packages.pasqal_adapters.pulser_adapter import build_simple_sequence_summary

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = get_logger(__name__)


class RLSequenceAgent(BaseAgent):
    """Generate pulse-sequence candidates using a trained PPO policy."""

    agent_name = AgentName.SEQUENCE

    def __init__(
        self,
        param_space: PhysicsParameterSpace | None = None,
        checkpoint_path: str | Path | None = None,
        n_candidates: int = 5,
        temperature: float = 0.3,
        enabled: bool = False,
        heuristic_agent: SequenceProtocol | None = None,
    ) -> None:
        super().__init__()
        self.param_space = param_space or PhysicsParameterSpace.default()
        self._policy = None
        self.n_candidates = n_candidates
        self.temperature = temperature
        self.enabled = enabled
        self.heuristic_agent: SequenceProtocol = heuristic_agent or SequenceAgent(param_space=self.param_space)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        if enabled and checkpoint_path and TORCH_AVAILABLE:
            self._load_policy(Path(checkpoint_path))
        elif enabled and not TORCH_AVAILABLE:
            logger.warning("RLSequenceAgent enabled but PyTorch not installed - fallback mode.")
            self.enabled = False

    @property
    def ready(self) -> bool:
        return bool(self.enabled and self._policy is not None)

    def _load_policy(self, path: Path) -> None:
        if not path.exists():
            logger.warning("PPO checkpoint not found at %s - fallback mode.", path)
            self.enabled = False
            return
        from packages.ml.ppo import ActorCritic

        self._policy = ActorCritic.from_checkpoint(path)
        self._policy.eval()
        logger.info("PPO policy loaded from %s", path)

    def _build_obs(self, spec: ExperimentSpec, register: RegisterCandidate) -> np.ndarray:
        from packages.ml.rl_env import PulseDesignEnv

        env = PulseDesignEnv(spec, [register], param_space=self.param_space, simulate_fn=lambda *_: 0.0)
        env.reset(seed=0)
        return env._build_observation()  # noqa: SLF001 - shared observation contract

    def run(
        self,
        spec: ExperimentSpec,
        register_candidate: RegisterCandidate,
        campaign_id: str,
        memory_records: list[MemoryRecord] | None = None,
    ) -> list[SequenceCandidate]:
        if not self.ready:
            return self.heuristic_agent.run(spec, register_candidate, campaign_id, memory_records)

        obs = self._build_obs(spec, register_candidate)
        candidates: list[SequenceCandidate] = []

        for i in range(self.n_candidates):
            action, _, _ = self._policy.get_action(obs, deterministic=self.temperature <= 0)
            params = rescale_action(action, self.param_space)
            family_enum = params.pop("family_enum")
            duration_ns = int(params["duration_ns"])

            seq = SequenceCandidate(
                campaign_id=campaign_id,
                spec_id=spec.id,
                register_candidate_id=register_candidate.id,
                label=f"{register_candidate.label}-rl-{params['family']}-{i}",
                sequence_family=family_enum,
                channel_id="rydberg_global",
                duration_ns=duration_ns,
                amplitude=float(params["amplitude"]),
                detuning=float(params["detuning"]),
                phase=0.0,
                waveform_kind="constant",
                predicted_cost=self.param_space.cost_for(register_candidate.atom_count, duration_ns),
                reasoning_summary=(
                    f"RL policy generated {params['family']} pulse "
                    f"Omega={params['amplitude']:.1f} rad/us, delta={params['detuning']:.1f} rad/us, "
                    f"T={duration_ns} ns for {register_candidate.label}."
                ),
                metadata={
                    "atom_count": register_candidate.atom_count,
                    "layout_type": register_candidate.layout_type,
                    "spacing_um": register_candidate.metadata.get("spacing_um"),
                    "source": "rl_policy",
                    "candidate_index": i,
                    "temperature": self.temperature,
                },
            )
            seq = seq.model_copy(
                update={
                    "serialized_pulser_sequence": build_simple_sequence_summary(
                        register_candidate,
                        seq,
                        param_space=self.param_space,
                    )
                }
            )
            candidates.append(seq)

        return candidates
