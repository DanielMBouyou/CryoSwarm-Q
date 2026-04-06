from __future__ import annotations

"""Verify that concrete agents satisfy their runtime protocols."""

import pytest

from packages.agents.campaign_agent import CampaignAgent
from packages.agents.geometry_agent import GeometryAgent
from packages.agents.memory_agent import MemoryAgent
from packages.agents.noise_agent import NoiseRobustnessAgent
from packages.agents.problem_agent import ProblemFramingAgent
from packages.agents.protocols import (
    AgentProtocol,
    CampaignProtocol,
    GeometryProtocol,
    MemoryCaptureProtocol,
    NoiseEvaluationProtocol,
    ProblemFramingProtocol,
    ResultsProtocol,
    RoutingProtocol,
    SequenceProtocol,
)
from packages.agents.results_agent import ResultsAgent
from packages.agents.routing_agent import BackendRoutingAgent
from packages.agents.sequence_agent import SequenceAgent


_PROTOCOL_MAP: list[tuple[type, type]] = [
    (ProblemFramingAgent, ProblemFramingProtocol),
    (GeometryAgent, GeometryProtocol),
    (SequenceAgent, SequenceProtocol),
    (NoiseRobustnessAgent, NoiseEvaluationProtocol),
    (BackendRoutingAgent, RoutingProtocol),
    (CampaignAgent, CampaignProtocol),
    (ResultsAgent, ResultsProtocol),
    (MemoryAgent, MemoryCaptureProtocol),
]


@pytest.mark.parametrize(
    ("agent_class", "protocol"),
    _PROTOCOL_MAP,
    ids=[pair[0].__name__ for pair in _PROTOCOL_MAP],
)
def test_agent_satisfies_protocol(agent_class: type, protocol: type) -> None:
    agent = agent_class()
    assert isinstance(agent, protocol), (
        f"{agent_class.__name__} does not satisfy {protocol.__name__}"
    )


@pytest.mark.parametrize(
    "agent_class",
    [pair[0] for pair in _PROTOCOL_MAP],
    ids=[pair[0].__name__ for pair in _PROTOCOL_MAP],
)
def test_all_agents_satisfy_base_protocol(agent_class: type) -> None:
    agent = agent_class()
    assert isinstance(agent, AgentProtocol)
