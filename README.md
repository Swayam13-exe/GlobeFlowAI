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
- 🤖 **Inference Integration:** Built-in OpenAI-compatible inference runner for immediate evaluation.

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
workforce-mobility-env/
├── openenv.yaml        # OpenEnv metadata & task registry
├── Dockerfile          # FastAPI deployment config (HF Spaces port 7860)
├── main.py             # FastAPI application entry point
├── inference.py        # OpenAI-compatible inference runner
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

## ⚙️ Environment Design

### State
The `WorkforceState` is the single source of truth, tracking:
- **Employee Info:** Role and dependent status.
- **Documents:** Current status (`missing`, `submitted`, `verified`, `rejected`) and ground-truth validity.
- **Departments:** Approval flags for HR, Legal, and Finance.
- **Compliance:** Checklist for tax registration, payroll, PDPA, and shadow payroll.
- **Progress:** A normalized `[0.0, 1.0]` fraction of the total checklist completed.

### Actions
Agents interact using a discrete action space:
- `request_document` / `verify_document`: Manage the document lifecycle.
- `approve_hr` / `approve_legal` / `approve_finance`: Move through the approval chain.
- `set_payroll` / `set_tax_id` / `set_pdpa` / `set_shadow_payroll`: Configure country-specific compliance.
- `finalize_case`: Terminate the episode (valid only when all blockers are resolved).

---

## 🎯 Tasks

### 🟢 Easy: India → Germany (Score ceiling: 0.99)
- **Scenario:** Single relocation of an Engineer, no dependents.
- **Key Challenge:** Linear path — verify 4 docs, get HR + Legal approval, register tax/payroll.
- **Expected Score:** `0.70 – 1.00`
- **Perfect Agent Score:** ~`0.96 – 0.99`

### 🟡 Medium: India → Singapore (Score ceiling: 0.79)
- **Scenario:** Manager with dependents relocating to Singapore.
- **Key Challenge:** Compliance density — PDPA consent, shadow payroll, correct Employment Pass visa. Degree certificate is a **trap** (not required, attempting it costs parsimony points).
- **Expected Score:** `0.40 – 0.80`
- **Perfect Agent Score:** ~`0.75 – 0.79`

### 🔴 Hard: India → Germany + UAE (Score ceiling: 0.59)
- **Scenario:** Simultaneous relocation of a Director to **two** countries.
- **The Trap:** UAE has **NO income tax**. Calling `set_tax_id(UAE)` loses the 15% UAE compliance weight AND incurs an additional -0.10 penalty.
- **Expected Score:** `0.20 – 0.60`
- **Perfect Agent Score:** ~`0.55 – 0.59`

---

## 📊 Scoring Architecture (v2.0 — Calibrated)

### How Scores Are Calculated

| Stage | What Happens |
|-------|-------------|
| **1. Component Scoring** | Each required action contributes weighted points (docs, depts, compliance) |
| **2. Penalties** | Wrong visa type (-0.05), UAE tax called (-0.10 extra), junk actions (-0.03 each) |
| **3. Ceiling Clamp** | Score is clamped to the task-specific ceiling (`easy=0.99`, `medium=0.79`, `hard=0.59`) |
| **4. Range Check** | `finalize_case` compares this score with the expected bracket to determine `success`/`failed` |

### Why Ceilings?

The ceilings ensure that a **perfect agent** always lands **within** the expected score bracket:
```
Task    Bracket       Ceiling   Perfect Agent
──────  ────────────  ───────   ─────────────
easy    0.70 – 1.00   0.99      ~0.96–0.99
medium  0.40 – 0.80   0.79      ~0.75–0.79
hard    0.20 – 0.60   0.59      ~0.55–0.59
```

### Parsimony Penalty
Each **unnecessary action** (e.g., verifying `degree_certificate` for Singapore) costs `-0.03` in raw score, up to a maximum of `-0.15`. This rewards efficient agents.

---

## 💰 Step Reward System

| Action Result | Reward | Description |
|---------------|--------|-------------|
| **Success** | `+0.4` | Action resolved cleanly |
| **Milestone** | `+0.1` | Reached doc/dept/compliance group completion |
| **Done** | `+0.3` | `finalize_case` called successfully |
| **Wrong Action** | `-0.1` | Valid type, but wrong context |
| **Prereq Violation** | `-0.2` | Dependency order not respected |
| **Rule Violation** | `-0.3` | Violating country rules (e.g. UAE Tax ID) |
| **Invalid Action** | `-0.4` | Malformed or unknown action |

*All rewards are clamped to `[-1.0, 1.0]` per step.*

---

## 🌐 API Endpoints

All endpoints communicate using JSON with Pydantic v2 compliant schemas.

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

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/openenv-workforce
cd openenv-workforce

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate.bat     # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --host 0.0.0.0 --port 7860
```

---

## 🐳 Docker Usage

```bash
# Build the application
docker build -t openenv-workforce .

# Run the API server
docker run -p 7860:7860 openenv-workforce
```

---

## 🤗 Deploying to HuggingFace Spaces

### Step-by-Step Deployment

**1. Create a HuggingFace Space**

Go to [huggingface.co/new-space](https://huggingface.co/new-space) and:
- Set **Space name**: `openenv-workforce`
- Select **SDK**: `Docker`
- Set **Visibility**: Public (for OpenEnv evaluation) or Private

**2. Push your code using Git**

```bash
# Install git-lfs (required for HF)
git lfs install

# Clone your new Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/openenv-workforce
cd openenv-workforce

# Copy your project files into the cloned directory
# Then commit and push:
git add .
git commit -m "Initial deployment: OpenEnv Workforce Mobility v1.0"
git push
```

**3. Verify the Space is running**

Once pushed, HuggingFace will automatically build the Docker image using your `Dockerfile`. You can monitor the build log in the Space's **"Factory Build"** tab.

Check the API health:
```bash
curl https://YOUR_USERNAME-openenv-workforce.hf.space/
# Expected: {"status": "ok", "environment": "openenv-workforce"}
```

**4. Run inference against the deployed Space**

```bash
export API_BASE_URL="https://YOUR_USERNAME-openenv-workforce.hf.space"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="your-openai-key"

python inference.py
```

**5. Add secrets (optional — for private models)**

In your Space's **Settings → Secrets**, add:
- `OPENAI_API_KEY` — your API key
- `HF_TOKEN` — your HuggingFace token if using private models

---

## 🤖 Running the Inference Agent Locally

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"
export MODEL_NAME="gpt-4o-mini"        # or gpt-4o, gpt-3.5-turbo

# Run all three tasks in sequence
python inference.py
```

### Expected Output

```
Model:    gpt-4o-mini
API URL:  https://api.openai.com/v1
Max steps per task: 20

============================================================
  TASK: EASY
  Countries: ['Germany']
============================================================
  [step  1] request_document:passport    → success   reward=+0.40
  ...
  [step 13] finalize_case:all            → success   reward=+0.70

  Final status: success
  Final score:  0.9700

============================================================
  OPENENV-WORKFORCE — FINAL SCORES
============================================================
  Task         Score  Range             Status                 Bar
  -------------------------------------------------------
  easy         0.9700  [0.70–1.00]       ✓ success              [███████████████████░]
  medium       0.7500  [0.40–0.80]       ✓ success              [███████████████░░░░░]
  hard         0.5500  [0.20–0.60]       ✓ success              [███████████░░░░░░░░░]
  -------------------------------------------------------
  Average      0.7567
============================================================
```

---

## ✅ OpenEnv Compliance

`openenv-workforce` is strictly built to the **OpenEnv v1.0** specification:
- **Sessionized:** Each episode is isolated using a `session_id` UUID.
- **Deterministic:** All grading is pure Python — zero LLM dependency, 100% reproducible.
- **Standard Interface:** Follows `reset()`, `step()`, and `state()` interaction pattern.
- **Score-Driven Status:** `finalize_case` evaluates score vs. expected range, not just blocker presence.

---

## 📄 API Usage Example

### Reset
```bash
curl -X POST https://YOUR_USERNAME-openenv-workforce.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy"}'
```

### Step
```bash
curl -X POST https://YOUR_USERNAME-openenv-workforce.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID", "action": {"action_type": "request_document", "target": "passport"}}'
```

### State
```bash
curl "https://YOUR_USERNAME-openenv-workforce.hf.space/state?session_id=YOUR_SESSION_ID"
```

---

## 🔮 Future Improvements

- [ ] **Multi-Agent Mode:** Support for collaborative relocation cases (Agent A as HR, Agent B as Legal).
- [ ] **Dynamic Rules:** Injecting new tax treaties at runtime via fixture endpoints.
- [ ] **Extended Countries:** Adding support for US, UK, and India as destinations.
- [ ] **Persistent Sessions:** Redis-backed session storage for multi-process deployments.

---

## ✍️ Authors

- **Team AI Kalesh** - *Design & Implementation*

---

*Built as an OpenEnv-compliant evaluation environment. For evaluation framework documentation, see [OpenEnv specification](https://openenv.ai).*
