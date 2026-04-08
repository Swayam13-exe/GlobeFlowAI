---
title: OpenEnv Workforce Mobility
emoji: 🏢
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# GlobeFlowAI — Global Mobility & Compliance Orchestrator

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/Swayam14/openenv-workforce)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)

An OpenEnv-compatible reinforcement learning environment where an AI agent handles **real-world employee relocation cases** — processing documents, navigating multi-country compliance rules, and managing department approvals to finalize international transfers.

Built for the **Meta × Scaler OpenEnv AI Hackathon 2026**.

---

## What is GlobeFlowAI?

GlobeFlowAI simulates the full lifecycle of relocating an employee from **India** to one of three destination countries: **Germany**, **Singapore**, or **UAE**. The agent must learn to:

- Request and verify the correct documents for each country
- Obtain department approvals in the correct order (HR → Legal → Finance)
- Configure country-specific compliance items (tax registration, payroll, PDPA consent)
- Resolve multi-country rule conflicts (e.g. Germany requires tax ID, UAE does not allow it)
- Finalize the relocation case without errors

This environment models a genuine enterprise workflow that multinational companies deal with thousands of times per year — making it a rich, non-trivial domain for agent training and evaluation.

---

## Environment Design

### Architecture

```
GlobeFlowAI/
├── env/
│   ├── environment.py      # Core WorkforceEnv — reset/step/state
│   ├── models.py           # Pydantic v2 typed models
│   ├── validators.py       # Pure validation functions
│   ├── reward.py           # Shaped reward function
│   ├── tasks.py            # Task definitions (easy/medium/hard)
│   ├── rules.py            # Re-export of rules engine
│   ├── rules_engine.py     # Country rules, fixture loading
│   └── graders.py          # Shim → graders/graders.py
├── graders/
│   └── graders.py          # Deterministic task graders
├── server/
│   └── app.py              # FastAPI entry point
├── fixtures/
│   ├── country_rules.json  # Per-country compliance rules
│   ├── visa_types.json     # Visa type metadata
│   └── tax_treaties.json   # India bilateral tax treaties
├── main.py                 # FastAPI app with session management
├── inference.py            # OpenAI-powered baseline agent
├── openenv.yaml            # OpenEnv spec metadata
├── Dockerfile              # Container definition
└── requirements.txt        # Python dependencies
```

### State Design

The environment maintains a stateful `WorkforceState` with:

| Field | Type | Description |
|-------|------|-------------|
| `case_id` | str | Unique case identifier |
| `task_name` | str | easy / medium / hard |
| `employee` | EmployeeRecord | Role and dependent status |
| `countries` | list[str] | Destination countries (1–2) |
| `documents` | dict | Document name → status + validity |
| `departments` | dict | HR / Legal / Finance approval flags |
| `compliance` | dict | tax_id / payroll / pdpa / shadow_payroll |
| `conflicts` | list | Active rule conflicts (hard task) |
| `deadline_days` | int | Steps remaining before auto-fail |
| `progress` | float | Completion fraction [0.0, 1.0] |
| `status` | str | in_progress / success / failed |

---

## Action Space

| Action Type | Target | Description |
|-------------|--------|-------------|
| `request_document` | document name | Submit a document for verification |
| `verify_document` | document name | Verify a submitted document |
| `approve_hr` | *(empty)* | HR department approval |
| `approve_legal` | *(empty)* | Legal approval (requires all docs verified) |
| `approve_finance` | *(empty)* | Finance approval (requires Legal + no conflicts) |
| `set_payroll` | *(empty)* | Configure host-country payroll |
| `set_tax_id` | *(empty)* | Register tax ID (Germany only — **not UAE**) |
| `set_shadow_payroll` | *(empty)* | Enable shadow payroll (Singapore only) |
| `set_pdpa` | *(empty)* | Collect PDPA consent (Singapore only) |
| `resolve_conflict` | *(empty)* | Resolve a rule conflict (hard task) |
| `finalize_case` | *(empty)* | Close the case (all requirements must be met) |

**Action format:**
```json
{"action_type": "request_document", "target": "passport"}
{"action_type": "approve_hr", "target": ""}
```

---

## Observation Space

After every `reset()` and `step()`, the agent receives an `Observation` containing:

| Field | Description |
|-------|-------------|
| `state` | Full `WorkforceState` (documents, departments, compliance, conflicts) |
| `available_actions` | List of currently valid actions the agent can take |
| `current_blockers` | Reasons `finalize_case` cannot be called yet |
| `last_action_result` | Result code of the last action |
| `last_action_error` | Error detail if last action failed |
| `steps_taken` | Number of steps used this episode |
| `done` | True when episode has ended |

---

## Tasks

### Task 1 — Easy: India → Germany

**Scenario:** Relocate an Engineer from India to Germany via EU Blue Card pathway.

| Property | Value |
|----------|-------|
| Countries | Germany |
| Employee | Engineer, no dependents |
| Documents required | passport, visa, employment_letter, work_permit |
| Departments required | HR only |
| Compliance required | tax_id, payroll |
| Deadline | 20 steps |
| Max steps | 25 |
| Expected score range | 0.70 – 0.95 |

**Optimal sequence (~11 steps):**
```
request + verify (x4 docs) → approve_hr → set_tax_id → set_payroll → finalize_case
```

---

### Task 2 — Medium: India → Singapore

**Scenario:** Relocate a Manager with dependents from India to Singapore via Employment Pass.

| Property | Value |
|----------|-------|
| Countries | Singapore |
| Employee | Manager, has dependents |
| Documents required | passport, visa, employment_letter |
| Departments required | HR, Legal |
| Compliance required | payroll, pdpa, shadow_payroll |
| Deadline | 25 steps |
| Max steps | 25 |
| Expected score range | 0.40 – 0.75 |

**Key rules:**
- Singapore does **NOT** require tax_id — calling `set_tax_id` incurs a −0.30 penalty
- PDPA consent must be collected before finalization
- Shadow payroll is mandatory for home-country tax tracking
- Legal must approve before `finalize_case`

**Optimal sequence (~11 steps):**
```
request + verify (x3 docs) → approve_hr → approve_legal →
set_payroll → set_pdpa → set_shadow_payroll → finalize_case
```

---

### Task 3 — Hard: India → Germany + UAE

**Scenario:** Simultaneous relocation of a Director with dependents to both Germany and UAE.

| Property | Value |
|----------|-------|
| Countries | Germany, UAE |
| Employee | Director, has dependents |
| Documents required | passport, visa, employment_letter, work_permit |
| Departments required | HR, Legal, Finance |
| Compliance required | tax_id (Germany only), payroll |
| Deadline | 30 steps |
| Max steps | 25 |
| Expected score range | 0.20 – 0.60 |

**Critical trap — UAE has NO income tax:**
> ⚠️ Calling `set_tax_id` with target `UAE` is a **rule violation** that incurs a −0.30 reward penalty AND reduces the final grader score significantly. Agents must learn to call `set_tax_id` for Germany ONLY.

**Key rules:**
- A `tax_conflict` exists between Germany (requires tax_id) and UAE (no income tax)
- `resolve_conflict` must be called **before** `approve_finance`
- Finance cannot approve while unresolved conflicts remain
- All 4 documents must be verified before Legal can approve

**Optimal sequence (~14 steps):**
```
request + verify (x4 docs) → approve_hr → approve_legal →
set_tax_id (Germany) → set_payroll → resolve_conflict →
approve_finance → finalize_case
```

---

## Reward Function

### Per-Step Rewards

| Event | Reward |
|-------|--------|
| Successful action | +0.30 |
| Document verified (milestone) | +0.20 bonus |
| Department approved (milestone) | +0.20 bonus |
| Compliance item set (milestone) | +0.15 bonus |
| Conflict resolved (milestone) | +0.25 bonus |
| Episode finalized successfully | +0.50 bonus |
| Progress increase | +0.05 × delta |
| Wrong action (valid type, wrong context) | −0.10 |
| Prerequisite violated | −0.20 |
| Rule violation (e.g. UAE tax) | −0.30 |
| Invalid action (unknown type) | −0.30 |
| Repeated action | −0.10 |

All per-step rewards are clamped to `[−1.0, 1.0]`.

### Partial Progress Shaping

The reward function provides dense signal throughout the episode — not just at the end. Progress is computed as a weighted sum:

- Documents: 40%
- Departments: 35%
- Compliance: 15%
- Conflict resolution: 10%

This means agents receive gradient signal even when they never reach `finalize_case`.

---

## Grader System

Each task has a deterministic grader that returns a score strictly in `(0.0, 1.0)` — exclusive bounds required by the OpenEnv validator.

### Easy Grader Weights
| Component | Weight |
|-----------|--------|
| Documents verified (4 docs) | 0.40 |
| HR approved | 0.25 |
| Compliance done (tax_id + payroll) | 0.25 |
| Case finalized | 0.10 |
| Ceiling | 0.949 |

### Medium Grader Weights
| Component | Weight |
|-----------|--------|
| Documents verified (3 docs) | 0.30 |
| HR approved | 0.15 |
| Legal approved | 0.20 |
| Compliance done (payroll + pdpa + shadow) | 0.25 |
| Case finalized | 0.10 |
| Ceiling | 0.749 |

### Hard Grader Weights
| Component | Weight |
|-----------|--------|
| Documents verified (4 docs) | 0.25 |
| HR approved | 0.08 |
| Legal approved | 0.08 |
| Finance approved | 0.08 |
| Compliance done (tax_id + payroll) | 0.16 |
| Conflict resolved | 0.15 |
| Case finalized | 0.10 |
| UAE tax violation penalty | −0.25 |
| Ceiling | 0.599 |

---

## HTTP API

The environment runs as a FastAPI server on port 7860.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check — returns `{"status": "ok"}` |
| GET | `/health` | Health check (explicit) |
| GET | `/tasks` | List available tasks |
| POST | `/reset` | Start new episode `{"task_name": "easy"}` |
| POST | `/step` | Apply action `{"action_type": "...", "target": "..."}` |
| GET | `/state` | Current state (query param `?session_id=...`) |
| POST | `/grade` | Get current grader score |

### Example

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "request_document", "target": "passport"}'

# Grade
curl -X POST http://localhost:7860/grade \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- Docker
- OpenAI API key

### Local Installation

```bash
# Clone the repo
git clone https://github.com/Swayam14/openenv-workforce
cd openenv-workforce

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_openai_api_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini

# Start the server
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Run the Baseline Agent

```bash
python inference.py
```

### Run Tests

```bash
python test_eval.py
```

### Docker

```bash
# Build
docker build -t globeflowai .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=your_openai_api_key \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  globeflowai
```

---

## Baseline Scores

Running `python inference.py` with `gpt-4o-mini`:

```
EASY     | score=0.780 | steps=14 | success
MEDIUM   | score=0.750 | steps=13 | success
HARD     | score=0.600 | steps=15 | success

Average Score: 0.710
```

### Score Interpretation

| Task | Score Range | Difficulty | Why |
|------|-------------|------------|-----|
| Easy | 0.70 – 0.95 | Low | Single country, only HR needed, no compliance traps |
| Medium | 0.40 – 0.75 | Medium | Singapore-specific rules, PDPA + shadow payroll |
| Hard | 0.20 – 0.60 | High | Multi-country conflict, UAE tax trap, Finance required |

Scores decrease with task difficulty, reflecting the increasing number of rules the agent must learn and the cost of mistakes like calling `set_tax_id` for UAE (−0.30 reward + grader penalty).

---

## Country Rules Summary

| Rule | Germany | Singapore | UAE |
|------|---------|-----------|-----|
| Visa required | ✓ | ✓ | ✓ |
| Tax ID required | ✓ | ✗ | ✗ (**no income tax**) |
| Payroll required | ✓ | ✓ | ✓ |
| PDPA consent | ✗ | ✓ | ✗ |
| Shadow payroll | ✗ | ✓ | ✗ |
| Finance approval | ✓ | ✗ | ✓ |

---

## OpenEnv Spec Compliance

- ✅ `reset()` → returns typed `Observation`
- ✅ `step(action)` → returns typed `StepResult` with `observation`, `reward`, `done`, `info`
- ✅ `state()` → returns typed `WorkforceState`
- ✅ Pydantic v2 typed models throughout
- ✅ `openenv.yaml` with task registry
- ✅ Dockerfile builds and runs cleanly
- ✅ FastAPI server on port 7860
- ✅ `/health` endpoint responds to HuggingFace Space ping
- ✅ All grader scores strictly between 0 and 1 (exclusive)
- ✅ `inference.py` uses OpenAI client with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- ✅ `[START]` / `[STEP]` / `[END]` stdout logging format

---

## Team

**Team AI Kalesh**

Built for the Meta × Scaler OpenEnv AI Hackathon — India 2026.

---

## License

This project is licensed under the MIT License.