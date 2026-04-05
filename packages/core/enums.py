from __future__ import annotations

from enum import StrEnum


class AppEnvironment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class GoalStatus(StrEnum):
    DRAFT = "draft"
    STORED = "stored"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CampaignStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CandidateStatus(StrEnum):
    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    RANKED = "ranked"
    REJECTED = "rejected"


class BackendType(StrEnum):
    LOCAL_PULSER_SIMULATION = "local_pulser_simulation"
    EMU_SV_CANDIDATE = "emu_sv_candidate"
    EMU_MPS_CANDIDATE = "emu_mps_candidate"


class AgentName(StrEnum):
    PROBLEM_FRAMING = "problem_framing_agent"
    GEOMETRY = "geometry_agent"
    SEQUENCE = "sequence_agent"
    NOISE = "noise_robustness_agent"
    ROUTING = "backend_routing_agent"
    CAMPAIGN = "campaign_agent"
    RESULTS = "results_agent"
    MEMORY = "memory_agent"


class SequenceFamily(StrEnum):
    GLOBAL_RAMP = "global_ramp"
    DETUNING_SCAN = "detuning_scan"
    ADIABATIC_SWEEP = "adiabatic_sweep"


class NoiseLevel(StrEnum):
    LOW = "low_noise"
    MEDIUM = "medium_noise"
    STRESSED = "stressed_noise"


class DecisionType(StrEnum):
    SPECIFICATION = "specification"
    CANDIDATE_GENERATION = "candidate_generation"
    ROBUSTNESS_EVALUATION = "robustness_evaluation"
    BACKEND_ROUTING = "backend_routing"
    CAMPAIGN_RANKING = "campaign_ranking"
    RESULTS_SUMMARY = "results_summary"
    MEMORY_CAPTURE = "memory_capture"
