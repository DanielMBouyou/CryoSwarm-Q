from __future__ import annotations

"""Tests for the CryoSwarm-Q exception hierarchy."""

import pickle

import pytest

from packages.core.exceptions import (
    AdapterError,
    AgentError,
    ConfigurationError,
    CryoSwarmError,
    EmulatorError,
    EvaluationError,
    GeometryError,
    HamiltonianError,
    ProblemFramingError,
    RepositoryError,
    SequenceError,
    SimulationError,
)


def test_hierarchy() -> None:
    assert issubclass(AgentError, CryoSwarmError)
    assert issubclass(SimulationError, CryoSwarmError)
    assert issubclass(RepositoryError, CryoSwarmError)
    assert issubclass(HamiltonianError, SimulationError)
    assert issubclass(ProblemFramingError, AgentError)


def test_agent_error_message_includes_name() -> None:
    err = ProblemFramingError("bad goal")
    assert "problem_framing_agent" in str(err)
    assert "bad goal" in str(err)


@pytest.mark.parametrize(
    "exc_class",
    [CryoSwarmError, AgentError, SimulationError, RepositoryError, ConfigurationError, AdapterError],
)
def test_exceptions_are_picklable(exc_class: type[Exception]) -> None:
    if exc_class is AgentError:
        exc = exc_class("test_agent", "test message")
    else:
        exc = exc_class("test message")
    restored = pickle.loads(pickle.dumps(exc))
    assert str(restored) == str(exc)


@pytest.mark.parametrize(
    "exc",
    [
        ProblemFramingError("bad framing"),
        GeometryError("bad geometry"),
        SequenceError("bad sequence"),
        EvaluationError("bad evaluation"),
        HamiltonianError("bad hamiltonian"),
        EmulatorError("bad emulator"),
    ],
)
def test_specialized_exceptions_roundtrip(exc: Exception) -> None:
    restored = pickle.loads(pickle.dumps(exc))
    assert type(restored) is type(exc)
    assert str(restored) == str(exc)
