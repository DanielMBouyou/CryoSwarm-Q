# CLAUDE.md

## Project
CryoSwarm-Q

## One-sentence description
CryoSwarm-Q is a hardware-aware multi-agent software system for neutral-atom quantum experimentation, designed to generate, evaluate, rank, and orchestrate experiment candidates across Pulser and Pasqal-oriented workflows.

---

## Core objective
This repository must produce a research-grade software prototype for autonomous experiment design in neutral-atom quantum computing.

The system must:
- take a structured experimental objective as input,
- generate candidate neutral-atom registers and pulse sequences,
- evaluate candidates under noise and hardware constraints,
- choose the most relevant simulation or execution path,
- rank candidates according to robustness and feasibility,
- maintain a memory of past evaluations.

This is not a generic AI assistant project.
This is not a chatbot project.
This is not a generic scheduler.
This is a specialized agent-based orchestration layer for neutral-atom experimentation.

---

## Product vision
CryoSwarm-Q should feel like an autonomous research software layer sitting between:
- human scientific intent,
- pulse-level neutral-atom programming,
- simulation and emulator backends,
- and future execution workflows.

The repository should be useful for:
- hackathon demos,
- portfolio presentation,
- technical review by quantum software researchers,
- future extension into a more serious research prototype.

---

## Main technical pillars
The project must stay centered on these pillars:

1. **Neutral-atom hardware awareness**
   - geometry constraints
   - device constraints
   - pulse-level control representation
   - noise-aware candidate evaluation

2. **Multi-agent orchestration**
   - specialized agents
   - explicit responsibilities
   - non-monolithic decision logic
   - traceable decisions

3. **Simulation-first workflow**
   - candidate generation
   - robustness evaluation
   - emulator/backend routing
   - campaign-style ranking

4. **Research credibility**
   - clean abstractions
   - mathematically motivated scoring
   - reproducible experiments
   - explainable architecture

---

## Tools and ecosystem
The implementation should prioritize compatibility or conceptual alignment with:
- Python
- Pulser
- Pasqal Cloud SDK
- QoolQit
- PyTorch
- ROCm / AMD MI300X
- FastAPI
- Streamlit

Secondary optional tools:
- Ray
- SQLite or MongoDB
- Sentry
- GitHub Pages

---

## What to optimize for
When writing code or documentation for this project, optimize for:
- clarity
- modularity
- technical credibility
- demo-readiness
- traceability of agent decisions
- ease of extension

---

## What to avoid
Do not turn the project into:
- a generic LLM wrapper
- a chatbot interface
- a vague AI assistant
- a purely visual demo without logic
- a physics-heavy simulator with no orchestration value
- a fake "quantum" project disconnected from Pulser-style workflows

---

## Code style
- Prefer explicit and modular Python
- Use typed functions where reasonable
- Keep domain objects clear and documented
- Separate orchestration logic from simulation logic
- Keep adapters isolated from core agent logic
- Use simple interfaces between agents
- Favor correctness and readability over premature complexity

---

## Documentation style
All documentation must be:
- technically precise
- concise but serious
- written in clear English
- oriented toward deeptech / research software readers

Use vocabulary such as:
- neutral-atom experiment
- pulse sequence
- robustness
- candidate generation
- hardware-aware constraints
- orchestration
- campaign
- emulator routing
- experiment memory

Avoid vague language like:
- revolutionary
- magical
- disruptive
- genius
- plug-and-play quantum AI

---

## Deliverable target
First functional deliverable planned for May 2026.

The first deliverable should include:
- agent architecture
- candidate generation pipeline
- scoring and robustness layer
- backend/emulator routing
- campaign ranking
- visual dashboard
- repository documentation
- GitHub project page support

---

## Expected architecture
High-level modules should include:
- agents
- orchestration
- simulation
- scoring
- pasqal adapters
- memory
- dashboard
- api

---

## Success criteria
A strong milestone is reached if the repository can:
1. accept an experiment goal,
2. generate several candidate configurations,
3. evaluate them under constraints,
4. rank them meaningfully,
5. expose the reasoning in a readable way.

---

## Tone for generated content
When generating code, README content, page copy, or project descriptions:
- sound like a serious research software engineer
- sound grounded
- sound technically mature
- never overclaim collaboration or access to proprietary systems
