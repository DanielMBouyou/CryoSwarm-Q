# CryoSwarm-Q

CryoSwarm-Q is a hardware-aware multi-agent research software prototype for neutral-atom experimentation. It accepts a structured experiment goal, proposes register and pulse-sequence candidates, builds real Pulser sequences on a public analog device model, evaluates them with `QutipEmulator`, measures physically meaningful observables under noise and feasibility constraints, recommends an evaluation backend, ranks the campaign, stores traceable decisions in MongoDB Atlas, and exposes the workflow through FastAPI and Streamlit.

## Architecture Overview

The prototype is organized as a modular research stack:

- `packages/core`: typed models, enums, configuration, and logging
- `packages/db`: MongoDB connection, initialization, and repositories
- `packages/agents`: explicit agents for framing, geometry, sequence generation, robustness, routing, campaign ranking, results, and memory
- `packages/scoring`: objective, robustness, and ranking logic
- `packages/simulation`: deterministic synthetic evaluators and noise profiles
- `packages/pasqal_adapters`: Pulser integration plus optional adapters for QoolQit and Pasqal Cloud
- `packages/orchestration`: end-to-end pipeline and demo runner
- `apps/api`: FastAPI backend
- `apps/dashboard`: Streamlit dashboard
- `scripts`: connection checks and demo entrypoints
- `tests`: initial validation suite

## Agent Stack

CryoSwarm-Q currently includes these explicit agents:

- `ProblemFramingAgent`: converts a research goal into a structured experiment specification
- `GeometryAgent`: proposes plausible neutral-atom register layouts
- `SequenceAgent`: generates pulse-level control candidates compatible with Pulser-oriented workflows
- `NoiseRobustnessAgent`: evaluates nominal and perturbed candidate quality from actual Pulser emulation outputs
- `BackendRoutingAgent`: recommends `local_pulser_simulation`, `emu_sv_candidate`, or `emu_mps_candidate`
- `CampaignAgent`: ranks the full portfolio and updates campaign state
- `ResultsAgent`: builds the readable campaign summary
- `MemoryAgent`: stores reusable lessons from prior runs

## Data and Persistence

MongoDB Atlas is used for campaign persistence and agent traceability. The prototype stores:

- experiment goals
- register candidates
- sequence candidates
- robustness reports
- campaigns
- agent decisions
- memory records
- evaluation results

All identifiers are generated in the application layer to keep serialization explicit and easy to inspect.

## Environment Setup

Required environment variables are defined in [`.env.example`](/c:/Users/danie/Documents/CRYOSWARRM-Q/.env.example):

- `MONGODB_URI`
- `MONGODB_DB`
- `APP_ENV`
- `LOG_LEVEL`

Optional placeholders are included for future Pasqal Cloud integration:

- `PASQAL_CLOUD_USERNAME`
- `PASQAL_CLOUD_PASSWORD`
- `PASQAL_CLOUD_PROJECT_ID`

The base prototype does not require Pasqal credentials.

## Install

```powershell
py -m pip install -r requirements.txt
```

## Local Run

Initialize and test MongoDB:

```powershell
py scripts/test_mongo.py
```

Seed one goal manually:

```powershell
py scripts/seed_demo_goal.py
```

Run the full demo pipeline:

```powershell
py scripts/run_demo_pipeline.py
```

## FastAPI Backend

Start the API:

```powershell
py -m uvicorn apps.api.main:app --reload
```

Useful routes:

- `GET /health`
- `POST /goals`
- `GET /goals/{goal_id}`
- `POST /campaigns/run-demo`
- `GET /campaigns/{campaign_id}`
- `GET /campaigns/{campaign_id}/candidates`

## Streamlit Dashboard

Start the dashboard:

```powershell
py -m streamlit run apps/dashboard/app.py
```

The dashboard provides:

- project overview
- demo goal form
- latest campaigns
- ranked candidates
- agent decisions
- robustness reports

## Pasqal Tooling Note

This repository is designed to be compatible in spirit with public Pulser and Pasqal-oriented workflows. It does not claim access to proprietary Pasqal systems.

Current adapter posture:

- `Pulser`: used for public-device register validation, real `Sequence` construction, and emulator-backed observables
- `QoolQit`: optional adapter layer, installable locally, still isolated behind a conservative interface
- `Pasqal Cloud`: safe placeholder adapter with mock-safe behavior unless credentials are present

## Current Limitations

- the physical core now uses small-system Pulser emulation, but it is still a prototype rather than a calibrated hardware model
- robustness uses real Pulser noise parameters, but not a lab-calibrated experimental noise identification workflow
- backend routing is still rule-based, even though it now uses physical scale and robustness metrics
- Pasqal Cloud live submission remains intentionally disabled by default
- the dashboard is a first inspection surface, not a production UI

## Roadmap

Near-term roadmap:

- deepen Pulser-backed sequence construction and serialization
- add richer device-constraint modeling for neutral-atom geometries
- improve robustness estimators with more explicit perturbation families
- expand campaign memory retrieval and reuse
- add richer API endpoints for decision traces and robustness analytics
- harden the optional Pasqal Cloud adapter for real authenticated submission workflows

## Testing

Run the initial tests with:

```powershell
py -m pytest
```
