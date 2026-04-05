# CryoSwarm-Q

CryoSwarm-Q is a hardware-aware multi-agent research runtime for neutral-atom experimentation. The project is designed to transform a structured scientific objective into a ranked experimental campaign across Pulser and Pasqal-oriented workflows.

## Overview

The repository targets a specific problem in neutral-atom quantum computing: the gap between a research idea and a robust, runnable experiment campaign. Writing a single pulse sequence is not the bottleneck. The bottleneck is exploring the space of feasible register geometries, pulse parameterizations, noise assumptions, emulator choices, and evaluation budgets in a structured and traceable way.

CryoSwarm-Q is intended to sit between:

- human scientific intent,
- pulse-level neutral-atom programming,
- local and cloud simulation backends,
- and future execution workflows on Pasqal-oriented infrastructure.

This is not a generic assistant or a generic orchestration demo. It is a research software layer for autonomous experiment design under hardware-aware constraints.

## Problem Statement

A researcher exploring neutral-atom experiments typically has to:

- choose a register geometry,
- respect device and channel constraints,
- define one or more pulse strategies,
- evaluate candidates under noise,
- decide whether to use local simulation, exact emulation, or scalable approximation,
- compare outcomes across multiple variants,
- and iterate until a robust candidate set emerges.

CryoSwarm-Q addresses that workflow as an autonomous campaign problem rather than a one-off scripting task.

## Core Objective

The system must:

- accept a structured experimental objective,
- generate candidate neutral-atom registers and pulse sequences,
- evaluate them under device and noise constraints,
- route each candidate to the most relevant backend or emulator path,
- rank the resulting candidates by robustness and feasibility,
- and retain an experiment memory of prior evaluations.

## Design Principles

The repository is organized around four principles:

1. Hardware-aware modeling
   Geometry, device limits, pulse representation, backend selection, and noise assumptions must be explicit.

2. Multi-agent orchestration
   The system should rely on specialized agents with narrow responsibilities rather than a single opaque controller.

3. Simulation-first workflow
   Candidate generation, robustness analysis, emulator routing, and campaign ranking come before any future execution integration.

4. Research credibility
   The architecture must be explainable, reproducible, and technically grounded enough for review by quantum software practitioners.

## Agent Architecture

The initial architecture is expected to include the following roles:

- `ProblemFramingAgent`
  Converts a vague scientific request into a structured search objective with priorities and constraints.

- `GeometryAgent`
  Proposes neutral-atom register layouts that remain compatible with hardware-aware device assumptions.

- `SequenceAgent`
  Generates pulse sequence candidates for a given register, including controllable amplitude, detuning, and phase trajectories.

- `NoiseRobustnessAgent`
  Evaluates sensitivity to perturbations and estimates which candidates remain useful under realistic noise assumptions.

- `BackendRoutingAgent`
  Chooses the most relevant evaluation path, such as local Pulser simulation, exact emulation, or scalable approximation.

- `CampaignAgent`
  Manages portfolio-level decisions across the whole search campaign: exploration, pruning, prioritization, and compute budget allocation.

- `ResultsAgent`
  Aggregates outputs into a readable ranking, failure analysis, and recommendation set.

- `MemoryAgent`
  Retains prior outcomes, promising candidate families, fragile geometries, and backend selection patterns.

## Experimental Workflow

The target end-to-end flow is:

1. Receive an experiment goal.
2. Frame the goal as a structured search problem.
3. Generate candidate geometries.
4. Generate candidate pulse sequences.
5. Evaluate candidates under hardware constraints and noise assumptions.
6. Route each candidate to an appropriate simulator or emulator.
7. Rank the campaign by utility, robustness, and feasibility.
8. Persist the reasoning and outcomes in experiment memory.

## Decision Logic

The ranking model is intentionally simple to explain:

```text
candidate score = scientific quality + robustness - compute cost - delay penalty
```

In practical terms, CryoSwarm-Q should prefer candidates that:

- produce a strong experimental signal,
- remain stable under noise and device imperfections,
- do not consume unnecessary simulation or execution resources,
- and can be evaluated in a reasonable campaign timeline.

This matters because the project is not looking for the most optimistic theoretical result. It is looking for the most usable experiment candidate under realistic conditions.

### Robustness, in plain language

Robustness means a candidate should still perform reasonably well when the environment is no longer ideal. Instead of trusting a single perfect run, the system evaluates the same candidate across multiple perturbation scenarios such as noise, parameter drift, or control variation.

The practical question is:

```text
If we perturb this experiment several times, does it remain credible?
```

That is the core of the robustness layer.

### Pulse-level representation

Each pulse candidate is described through three time-dependent controls:

- `Omega(t)`: pulse amplitude,
- `delta(t)`: detuning,
- `phi(t)`: phase.

This is important because CryoSwarm-Q does not rank abstract ideas only. It ranks concrete control strategies that can later be simulated, compared, and routed toward Pasqal-oriented workflows.

## Pasqal-Oriented Scope

CryoSwarm-Q is designed to align conceptually with the Pasqal ecosystem rather than replace it.

Primary alignment targets:

- Pulser for neutral-atom register and pulse sequence representation,
- Pasqal Cloud SDK for future batch submission and execution workflows,
- Pasqal emulators such as EMU-SV and EMU-MPS for backend routing logic,
- QoolQit for higher-level analog problem construction where relevant.

The project value is the orchestration layer above these building blocks: candidate generation, robust evaluation, routing, ranking, and memory.

## Proposed Repository Modules

The first functional repository structure should converge toward:

- `agents/`
- `orchestration/`
- `simulation/`
- `scoring/`
- `pasqal_adapters/`
- `memory/`
- `dashboard/`
- `api/`

## Technology Direction

Primary ecosystem choices:

- Python
- Pulser
- Pasqal Cloud SDK
- QoolQit
- PyTorch
- FastAPI
- Streamlit
- ROCm / AMD MI300X

Optional supporting choices:

- Ray for distributed search or campaign execution,
- SQLite or MongoDB for experiment memory,
- GitHub Pages for project-facing documentation,
- Sentry for runtime monitoring.

## MongoDB Atlas

The repository is configured to support MongoDB Atlas through environment variables only. Credentials must stay in `.env` and must never be committed.

Required variables:

- `MONGODB_URI`
- `MONGODB_DATABASE`
- `MONGODB_COLLECTION`

To test the connection and insert one document:

```powershell
py -m pip install -r requirements.txt
py scripts/test_mongo.py
```

The script loads `.env`, pings the Atlas deployment, and inserts one test document into the configured collection.

## Why AMD MI300X Matters

The AMD MI300X direction is relevant only if it accelerates the experimental search loop. The project should use GPU resources for parallel candidate evaluation, robustness sweeps, surrogate scoring, or branch-level search acceleration. It should not consume accelerator resources for superficial UI tasks.

## Deliverable Target

The first functional milestone is planned for May 2026. A credible first deliverable should include:

- a visible multi-agent architecture,
- a candidate generation pipeline,
- a scoring and robustness layer,
- backend or emulator routing,
- campaign-style ranking,
- a dashboard for inspection,
- and documentation that explains the system in research-grade terms.

## Success Criteria

A meaningful milestone is reached when the repository can:

1. accept an experiment objective,
2. generate several candidate configurations,
3. evaluate them under explicit constraints,
4. rank them with readable justification,
5. and expose enough trace data for a researcher to understand why one path was preferred over another.

## What This Repository Must Avoid

CryoSwarm-Q should not become:

- a generic LLM wrapper,
- a chatbot interface,
- a vague autonomous agent demo,
- a purely visual prototype with no orchestration depth,
- or a fake quantum project detached from Pulser-style workflows.

## References

The current design direction is informed by the Pasqal documentation ecosystem and the AMD Developer Cloud hackathon context referenced during project definition:

- Pasqal Documentation: https://docs.pasqal.com/
- Pulser: https://docs.pasqal.com/pulser/
- Programming a neutral-atom QPU: https://docs.pasqal.com/pulser/programming/
- QoolQit: https://docs.pasqal.com/qoolqit/
- Pulser Studio: https://docs.pasqal.com/pulserstudio/
- Pasqal Cloud: https://docs.pasqal.com/cloud/
- Pasqal emulators: https://docs.pasqal.com/cloud/emulators/
- AMD Developer Hackathon context: https://lablab.ai/ai-hackathons/amd-developer

## Status

This repository is currently at the foundation stage. The immediate goal is to establish the architecture, the language of the project, and the initial implementation scaffolding for a serious neutral-atom experiment orchestration prototype.
