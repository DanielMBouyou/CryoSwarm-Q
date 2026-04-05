from __future__ import annotations

from abc import ABC
from typing import Any

from packages.core.enums import AgentName, DecisionType
from packages.core.logging import get_logger
from packages.core.models import AgentDecision


class BaseAgent(ABC):
    agent_name: AgentName

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def build_decision(
        self,
        campaign_id: str,
        subject_id: str,
        decision_type: DecisionType,
        status: str,
        reasoning_summary: str,
        structured_output: dict[str, Any],
    ) -> AgentDecision:
        return AgentDecision(
            campaign_id=campaign_id,
            agent_name=self.agent_name,
            subject_id=subject_id,
            decision_type=decision_type,
            status=status,
            reasoning_summary=reasoning_summary,
            structured_output=structured_output,
        )
