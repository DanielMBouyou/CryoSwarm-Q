from __future__ import annotations

"""Domain-specific exception hierarchy for CryoSwarm-Q.

These exceptions separate expected domain failures from generic Python errors.
They allow orchestration, adapters, and tests to target precise failure classes
without relying on broad ``except Exception`` patterns.
"""


class CryoSwarmError(Exception):
    """Base exception for all CryoSwarm-Q domain errors."""


class AgentError(CryoSwarmError):
    """An agent failed during execution."""

    def __init__(self, agent_name: str, message: str) -> None:
        self.agent_name = agent_name
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"[{self.agent_name}] {self.message}"

    def __reduce__(self) -> tuple[type["AgentError"], tuple[str, str]]:
        return type(self), (self.agent_name, self.message)


class ProblemFramingError(AgentError):
    """Problem framing agent could not produce a specification."""

    def __init__(self, message: str) -> None:
        super().__init__("problem_framing_agent", message)

    def __reduce__(self) -> tuple[type["ProblemFramingError"], tuple[str]]:
        return type(self), (self.message,)


class GeometryError(AgentError):
    """Geometry agent produced invalid candidates."""

    def __init__(self, message: str) -> None:
        super().__init__("geometry_agent", message)

    def __reduce__(self) -> tuple[type["GeometryError"], tuple[str]]:
        return type(self), (self.message,)


class SequenceError(AgentError):
    """Sequence generation failed."""

    def __init__(self, message: str) -> None:
        super().__init__("sequence_agent", message)

    def __reduce__(self) -> tuple[type["SequenceError"], tuple[str]]:
        return type(self), (self.message,)


class EvaluationError(AgentError):
    """Robustness evaluation or scoring failed."""

    def __init__(self, message: str) -> None:
        super().__init__("noise_robustness_agent", message)

    def __reduce__(self) -> tuple[type["EvaluationError"], tuple[str]]:
        return type(self), (self.message,)


class SimulationError(CryoSwarmError):
    """Raised when a physics simulation fails."""


class HamiltonianError(SimulationError):
    """Raised when Hamiltonian construction or diagonalization fails."""


class EmulatorError(SimulationError):
    """Raised when a backend emulator fails."""


class RepositoryError(CryoSwarmError):
    """Raised when database operations fail."""


class ConfigurationError(CryoSwarmError):
    """Raised when a configuration value is invalid or missing."""


class AdapterError(CryoSwarmError):
    """Raised when a Pasqal or Pulser adapter fails."""
