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

# 🚀 openenv-workforce

**A deterministic, OpenEnv-compliant simulation environment for AI workforce mobility compliance agents.**

---

## 📖 Overview

`openenv-workforce` is a specialized simulation environment for workforce mobility automation. Relocating an employee between countries is a high-stakes process involving multi-department approvals (HR, Legal, Finance) and strict country-specific compliance rules.

This environment challenges AI agents to act as **"Workforce Solutions Architects"**, making decisions that directly impact relocation timelines, costs, and legal compliance. It features a realistic **UAE no-tax trap** to test whether agents blindly follow patterns or strictly adhere to destination-specific rules.

---

## ✨ Features

- 🏢 **Multi-Department Workflow:** Simulates real corporate dependency chains (HR → Legal → Finance).
- 🌍 **Global Rule Engine:** Deterministic rules for Germany, Singapore, and UAE covering tax, visas, and PDPA.
- 🎯 **Three Complexity Tiers:** Progressive tasks from simple relocation to multi-country synchronization.
- ⚖️ **Calibrated Grading System:** Task ceilings ensure scores fall within expected brackets automatically.
- 🧩 **Parsimony Penalties:** Agents lose points for unnecessary actions, rewarding efficient execution.
- 🐳 **Deployment Ready:** Docker-optimized for HuggingFace Spaces and OpenEnv-compliant HTTP interfaces.
- 🤖 **Inference Integration:** Built-in OpenAI-compatible inference runner with structured stdout logging.

---

## 🔧 Environment Configuration Variables

The following variables **must** be set before running inference:

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | The API endpoint for the LLM (OpenAI-compatible) | `https://api.openai.com/v1` |
| `MODEL_NAME` | The model identifier to use for inference | `gpt-4o` |
| `HF_TOKEN` | Your Hugging Face / API key (takes precedence over `OPENAI_API_KEY`) | — |
| `OPENAI_API_KEY` | Standard OpenAI API key (used if `HF_TOKEN` is not set) | — |

Set them in your shell before running:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="hf_your_token_here"
```

Or add them as **Secrets** in your HuggingFace Space settings.

---

## 🏗️ System Architecture

The system follows a standard **Agent-Environment** loop exposed via a REST API:

1. **Environment (`WorkforceEnv`):** Maintains the in-memory state of employee records, document statuses, and compliance flags.
2. **Action Dispatcher:** Validates and executes agent actions (e.g., `verify_document`, `approve_hr`), updating state and calculating rewards.
3. **Reward Engine:** Clamps per-step rewards to `[-1.0, 1.0]` based on action results and checklist progress.
4. **Grader:** Evaluates final state and computes score clamped to task-specific ceiling.
5. **API Layer (FastAPI):** Exposes the `reset`, `step`, and `state` endpoints for remote agent interaction.

---

## 📁 Project Structure

```text
openenv-workforce/
├── openenv.yaml        # OpenEnv metadata & task registry
├── Dockerfile          # FastAPI deployment config (HF Spaces port 7860)
├── main.py             # FastAPI application entry point
├── inference.py        # OpenAI-compatible inference runner (root directory)
├── requirements.txt    # Pinned dependencies
├── env/
│   ├── environment.py  # Core WorkforceEnv logic
│   ├── models.py       # Pydantic v2 schemas (Source of Truth)
│   ├── rules.py        # Re-export shim for rules_engine
│   ├── rules_engine.py # Comprehensive country-rule logic
│   ├── reward.py       # Per-step reward & milestone bonuses
│   ├── tasks.py        # Task state fixtures & metadata
│   └── validators.py   # Pure validation functions
├── graders/
│   └── graders.py      # Calibrated deterministic task graders
└── fixtures/           # Rule data (Visa/Tax/Compliance)
    ├── country_rules.json
    ├── tax_treaties.json
    └── visa_types.json
```

---

## ⚙️ Observation Space

Each call to `/reset` or `/step` returns an **Observation** object with the following fields:

| Field | Type | Description |
|---|---|---|
| `state` | `WorkforceState` | Full environment state (see below) |
| `available_actions` | `list[str]` | Actions valid in the current state |
| `current_blockers` | `list[str]` | Reasons the case cannot be finalized yet |
| `last_action_result` | `str` | Result code of the last action taken |
| `last_action_error` | `str` | Error message if last action failed |
| `steps_taken` | `int` | Number of steps taken so far |

### WorkforceState Fields

| Field | Type | Description |
|---|---|---|
| `countries` | `list[str]` | Destination countries for relocation |
| `employee_role` | `str` | Role of the employee being relocated |
| `has_dependents` | `bool` | Whether employee has dependents |
| `documents` | `dict` | Document name → `{status, is_valid}` |
| `departments` | `dict` | Department name → `bool` (approved or not) |
| `compliance` | `dict` | Compliance item → `bool` (completed or not) |
| `progress` | `float` | Normalized `[0.0, 1.0]` checklist completion |
| `status` | `str` | Episode status: `in_progress`, `success`, `failed` |
| `deadline_days` | `int` | Days remaining before deadline |

---

## 🎮 Action Space

Agents interact using a **discrete action space** defined as `{action_type, target}` pairs:

| Action Type | Valid Targets | Description |
|---|---|---|
| `request_document` | `passport`, `visa`, `employment_letter`, `degree_certificate` | Request a document for submission |
| `verify_document` | `passport`, `visa`, `employment_letter`, `degree_certificate` | Verify a submitted document |
| `approve_hr` | `HR` | Grant HR department approval |
| `approve_legal` | `Legal` | Grant Legal department approval (requires all docs verified) |
| `approve_finance` | `Finance` | Grant Finance approval (requires Legal approval first) |
| `set_payroll` | `Germany`, `Singapore`, `UAE` | Register payroll for destination country |
| `set_tax_id` | `Germany`, `Singapore` | Register tax ID (**never UAE** — UAE has no income tax) |
| `set_shadow_payroll` | `Singapore` | Configure shadow payroll (Singapore only) |
| `set_pdpa` | `Singapore` | Set PDPA consent (Singapore only) |
| `finalize_case` | `all` | Complete the episode (valid only when all blockers resolved) |

---

## 🎯 Tasks

### 🟢 Easy: India → Germany (Score ceiling: 0.99)
- **Scenario:** Single relocation of an Engineer, no dependents.
- **Key Challenge:** Linear path — verify 4 docs, get HR + Legal approval, register tax/payroll.
- **Expected Score:** `0.70 – 1.00`
- **Perfect Agent Score:** ~`0.96 – 0.99`
- **Difficulty:** Easy

### 🟡 Medium: India → Singapore (Score ceiling: 0.79)
- **Scenario:** Manager with dependents relocating to Singapore.
- **Key Challenge:** Compliance density — PDPA consent, shadow payroll, correct Employment Pass visa. Degree certificate is a **trap** (not required, attempting it costs parsimony points).
- **Expected Score:** `0.40 – 0.80`
- **Perfect Agent Score:** ~`0.75 – 0.79`
- **Difficulty:** Medium

### 🔴 Hard: India → Germany + UAE (Score ceiling: 0.59)
- **Scenario:** Simultaneous relocation of a Director to **two** countries.
- **The Trap:** UAE has **NO income tax**. Calling `set_tax_id(UAE)` loses the 15% UAE compliance weight AND incurs an additional -0.10 penalty.
- **Expected Score:** `0.20 – 0.60`
- **Perfect Agent Score:** ~`0.55 – 0.59`
- **Difficulty:** Hard

---

## 📊 Baseline Scores

| Task | Expected Range | Perfect Agent | Difficulty |
|------|---------------|--------------|------------|
| Easy (India → Germany) | 0.70 – 1.00 | ~0.96 – 0.99 | 🟢 Easy |
| Medium (India → Singapore) | 0.40 – 0.80 | ~0.75 – 0.79 | 🟡 Medium |
| Hard (India → Germany + UAE) | 0.20 – 0.60 | ~0.55 – 0.59 | 🔴 Hard |

### Scoring Architecture

| Stage | What Happens |
|-------|-------------|
| **1. Component Scoring** | Each required action contributes weighted points (docs, depts, compliance) |
| **2. Penalties** | Wrong visa type (-0.05), UAE tax called (-0.10 extra), junk actions (-0.03 each) |
| **3. Ceiling Clamp** | Score is clamped to task-specific ceiling (`easy=0.99`, `medium=0.79`, `hard=0.59`) |
| **4. Range Check** | `finalize_case` compares score with expected bracket to determine `success`/`failed` |

### Step Reward System

| Action Result | Reward | Description |
|---------------|--------|-------------|
| **Success** | `+0.4` | Action resolved cleanly |
| **Milestone** | `+0.1` | Reached doc/dept/compliance group completion |
| **Done** | `+0.3` | `finalize_case` called successfully |
| **Wrong Action** | `-0.1` | Valid type, but wrong context |
| **Prereq Violation** | `-0.2` | Dependency order not respected |
| **Rule Violation** | `-0.3` | Violating country rules (e.g. UAE Tax ID) |
| **Invalid Action** | `-0.4` | Malformed or unknown action |

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/reset` | POST | Start a new episode: `{"task_name": "easy"}` |
| `/step` | POST | Submit action: `{"session_id": "...", "action": {"action_type": "...", "target": "..."}}` |
| `/state` | GET | Get current state: `?session_id=...` |

---

## 🛠️ Local Installation

### Prerequisites
- Python 3.11+
- pip
- Docker (for containerized deployment)

```bash
# Clone the repository
git clone https://huggingface.co/spaces/Swayam14/openenv-workforce
cd openenv-workforce

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate.bat     # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your_token_here"

# Start the server
uvicorn main:app --host 0.0.0.0 --port 7860
```

---

## 🐳 Docker Usage

```bash
# Build the image
docker build -t openenv-workforce .

# Run with environment variables
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o" \
  -e HF_TOKEN="your_token_here" \
  openenv-workforce
```

---

## 🤖 Running Inference

The inference script (`inference.py`) is in the root directory and emits structured logs in `[START]`, `[STEP]`, and `[END]` format:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your_token_here"

python inference.py
```

### Structured Log Format

Each event is a JSON line printed to stdout:

**[START]** — emitted at the beginning of each task:
```json
{"event": "START", "task": "easy", "countries": ["Germany"], "model": "gpt-4o", "api_base_url": "...", "max_steps": 20}
```

**[STEP]** — emitted after each action:
```json
{"event": "STEP", "task": "easy", "step": 1, "action_type": "request_document", "target": "passport", "result": "success", "reward": 0.4, "done": false, "progress": 0.1}
```

**[END]** — emitted at the end of each task:
```json
{"event": "END", "task": "easy", "final_score": 0.97, "final_status": "success", "steps_taken": 13}
```

---

## ✅ OpenEnv Compliance

`openenv-workforce` is strictly built to the **OpenEnv v1.0** specification:
- **Sessionized:** Each episode is isolated using a `session_id` UUID.
- **Deterministic:** All grading is pure Python — zero LLM dependency, 100% reproducible.
- **Standard Interface:** Follows `reset()`, `step()`, and `state()` interaction pattern.
- **Score-Driven Status:** `finalize_case` evaluates score vs. expected range, not just blocker presence.

---

## ✍️ Authors

- **Team AI Kalesh** - *Design & Implementation*

---

*Built as an OpenEnv-compliant evaluation environment for the OpenEnv Hackathon.*