from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

from pydantic import BaseModel

from packages.core.config import Settings, get_settings
from packages.core.models import (
    AgentDecision,
    CampaignState,
    EvaluationResult,
    ExperimentGoal,
    MemoryRecord,
    RegisterCandidate,
    RobustnessReport,
    SequenceCandidate,
)
from packages.db.mongodb import get_database


ModelT = TypeVar("ModelT", bound=BaseModel)


class CryoSwarmRepository:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.database = get_database(self.settings)
        self.collections = {
            "goals": self.database["experiment_goals"],
            "register_candidates": self.database["register_candidates"],
            "sequence_candidates": self.database["sequence_candidates"],
            "robustness_reports": self.database["robustness_reports"],
            "campaigns": self.database["campaigns"],
            "agent_decisions": self.database["agent_decisions"],
            "memory": self.database["memory"],
            "evaluation_results": self.database["evaluation_results"],
        }

    def _upsert_model(self, collection_name: str, model: BaseModel) -> BaseModel:
        document = model.model_dump(mode="json")
        document["_id"] = document["id"]
        self.collections[collection_name].replace_one({"_id": model.id}, document, upsert=True)
        return model

    def _insert_many(self, collection_name: str, models: Iterable[BaseModel]) -> list[BaseModel]:
        stored: list[BaseModel] = []
        for model in models:
            stored.append(self._upsert_model(collection_name, model))
        return stored

    def _parse_document(self, document: dict[str, Any] | None, model_class: type[ModelT]) -> ModelT | None:
        if document is None:
            return None
        document.pop("_id", None)
        return model_class.model_validate(document)

    def create_goal(self, goal: ExperimentGoal) -> ExperimentGoal:
        return self._upsert_model("goals", goal)

    def get_goal(self, goal_id: str) -> ExperimentGoal | None:
        document = self.collections["goals"].find_one({"_id": goal_id})
        return self._parse_document(document, ExperimentGoal)

    def create_campaign(self, campaign: CampaignState) -> CampaignState:
        return self._upsert_model("campaigns", campaign)

    def get_campaign(self, campaign_id: str) -> CampaignState | None:
        document = self.collections["campaigns"].find_one({"_id": campaign_id})
        return self._parse_document(document, CampaignState)

    def update_campaign(self, campaign: CampaignState) -> CampaignState:
        return self._upsert_model("campaigns", campaign)

    def update_campaign_status(
        self,
        campaign_id: str,
        status: str,
        summary: str | None = None,
    ) -> CampaignState | None:
        campaign = self.get_campaign(campaign_id)
        if campaign is None:
            return None
        campaign.status = status
        campaign.summary = summary or campaign.summary
        return self.update_campaign(campaign)

    def insert_register_candidates(self, candidates: list[RegisterCandidate]) -> list[RegisterCandidate]:
        return self._insert_many("register_candidates", candidates)

    def insert_sequence_candidates(self, candidates: list[SequenceCandidate]) -> list[SequenceCandidate]:
        return self._insert_many("sequence_candidates", candidates)

    def insert_robustness_report(self, report: RobustnessReport) -> RobustnessReport:
        return self._upsert_model("robustness_reports", report)

    def insert_agent_decision(self, decision: AgentDecision) -> AgentDecision:
        return self._upsert_model("agent_decisions", decision)

    def insert_memory_record(self, record: MemoryRecord) -> MemoryRecord:
        return self._upsert_model("memory", record)

    def insert_evaluation_results(self, results: list[EvaluationResult]) -> list[EvaluationResult]:
        return self._insert_many("evaluation_results", results)

    def list_candidates_for_campaign(self, campaign_id: str) -> list[EvaluationResult]:
        cursor = self.collections["evaluation_results"].find({"campaign_id": campaign_id}).sort(
            [("final_rank", 1), ("objective_score", -1)]
        )
        return [
            EvaluationResult.model_validate({k: v for k, v in doc.items() if k != "_id"})
            for doc in cursor
        ]

    def list_latest_campaigns(self, limit: int = 10) -> list[CampaignState]:
        cursor = self.collections["campaigns"].find().sort("created_at", -1).limit(limit)
        return [
            CampaignState.model_validate({k: v for k, v in doc.items() if k != "_id"})
            for doc in cursor
        ]

    def list_agent_decisions(self, campaign_id: str) -> list[AgentDecision]:
        cursor = self.collections["agent_decisions"].find({"campaign_id": campaign_id}).sort(
            "created_at", 1
        )
        return [
            AgentDecision.model_validate({k: v for k, v in doc.items() if k != "_id"})
            for doc in cursor
        ]

    def list_robustness_reports(self, campaign_id: str) -> list[RobustnessReport]:
        cursor = self.collections["robustness_reports"].find({"campaign_id": campaign_id}).sort(
            "created_at", 1
        )
        return [
            RobustnessReport.model_validate({k: v for k, v in doc.items() if k != "_id"})
            for doc in cursor
        ]

    def list_memory_records(self, campaign_id: str) -> list[MemoryRecord]:
        cursor = self.collections["memory"].find({"campaign_id": campaign_id}).sort("created_at", 1)
        return [
            MemoryRecord.model_validate({k: v for k, v in doc.items() if k != "_id"})
            for doc in cursor
        ]

    def list_recent_memory(self, limit: int = 10) -> list[MemoryRecord]:
        cursor = self.collections["memory"].find().sort("created_at", -1).limit(limit)
        return [
            MemoryRecord.model_validate({k: v for k, v in doc.items() if k != "_id"})
            for doc in cursor
        ]
