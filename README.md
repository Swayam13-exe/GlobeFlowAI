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

# OpenEnv Workforce Mobility

**A deterministic simulation environment for AI workforce mobility compliance agents.**

---

## Environment Description and Motivation

Employee relocation between countries is a high-stakes business process involving multi-department approvals (HR, Legal, Finance) and strict country-specific compliance rules. A single mistake—like attempting to register a tax ID in a zero-tax jurisdiction—can derail the entire relocation timeline and expose the company to legal risks.

`openenv-workforce` challenges AI agents to act as **Workforce Solutions Architects**, making sequential decisions that directly impact relocation success. The environment features:

- **Multi-Department Workflow:** Simulates real corporate dependency chains where Legal cannot approve until all documents are verified, and Finance cannot approve until Legal signs off.
- **Country-Specific Rules:** Each destination country (Germany, Singapore, UAE) has unique visa, tax, and compliance requirements.
- **Critical Trap:** UAE has no income tax—agents must recognize this and avoid calling `set_tax_id` for UAE, despite the pattern working for other countries.
- **Progressive Difficulty:** Three tasks ranging from straightforward single-country relocation to complex multi-country coordination with rule conflicts.

This environment tests whether AI agents can:
1. Follow complex prerequisite chains
2. Distinguish between country-specific rules vs. universal patterns
3. Optimize for efficiency (avoiding unnecessary actions)
4. Handle ambiguous situations where the obvious action is wrong

---

## Action Space

Agents interact using discrete `{action_type, target}` pairs. All actions are validated against the current state before execution.

| Action Type | Valid Targets | Description | Prerequisites |
|-------------|---------------|-------------|---------------|
| `request_document` | `passport`, `visa`, `employment_letter`, `work_permit` | Request document submission | None |
| `verify_document` | `passport`, `visa`, `employment_letter`, `work_permit` | Verify submitted document | Document must be requested first |
| `approve_hr` | `""` (empty) | Grant HR department approval | None |
| `approve_legal` | `""` | Grant Legal approval | **All required documents verified** |
| `approve_finance` | `""` | Grant Finance approval | **Legal approved + conflicts resolved (hard task)** |
| `set_payroll` | `""` or country name | Register payroll system | None |
| `set_tax_id` | `""` or country name | Register tax ID | **NEVER use for UAE (no income tax)** |
| `set_shadow_payroll` | `""` | Configure shadow payroll | Singapore only |
| `set_pdpa` | `""` | Set PDPA consent | Singapore only |
| `resolve_conflict` | `""` | Resolve rule conflicts | Hard task only, before Finance approval |
| `finalize_case` | `""` or `"all"` | Complete the episode | All blockers resolved |

### Action Results

| Result Code | Reward | Meaning |
|-------------|--------|---------|
| `success` | +0.4 | Action executed successfully |
| `wrong_action` | -0.1 | Valid action type, wrong context (e.g., verifying unsubmitted doc) |
| `prereq_violated` | -0.2 | Dependency order broken (e.g., Legal before docs verified) |
| `rule_violation` | -0.3 | Country rule broken (e.g., `set_tax_id:UAE`) |
| `invalid_action` | -0.4 | Malformed or unknown action |

---

## Observation Space

Each `/reset` or `/step` call returns an observation containing:

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `state` | `WorkforceState` | Complete environment state |
| `available_actions` | `list[str]` | Valid actions in current state |
| `current_blockers` | `list[str]` | Reasons preventing finalization |
| `last_action_result` | `str` | Result code of previous action |
| `last_action_error` | `str` | Error message if action failed |
| `steps_taken` | `int` | Number of actions taken |
| `done` | `bool` | Episode ended |

### WorkforceState Structure

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | `str` | Current task (`easy`, `medium`, `hard`) |
| `countries` | `list[str]` | Destination countries |
| `employee_role` | `str` | Employee's position |
| `has_dependents` | `bool` | Whether employee has family members |
| `documents` | `dict` | `{doc_name: {status, is_valid}}` |
| `departments` | `dict` | `{dept_name: approved_bool}` |
| `compliance` | `dict` | `{compliance_item: completed_bool}` |
| `conflicts` | `list[dict]` | Rule conflicts (hard task only) |
| `progress` | `float` | Checklist completion [0.0, 1.0] |
| `status` | `str` | `in_progress`, `success`, or `failed` |
| `deadline_days` | `int` | Days until relocation deadline |

---

## Task Descriptions with Expected Difficulty

### Easy: India → Germany (Single Country, Linear Path)

**Scenario:** Engineer relocating to Germany, no dependents.

**Required Actions:**
- **Documents:** Request and verify `passport`, `visa`, `employment_letter`, `work_permit` (4 docs)
- **Departments:** HR approval only
- **Compliance:** Register `tax_id` and `payroll` for Germany
- **Finalize:** Call `finalize_case` when all requirements met

**Difficulty Factors:**
- Linear dependency chain (straightforward order)
- Single destination country
- No special compliance rules
- Optimal path: ~11 steps

**Expected Score Range:** 0.70 – 1.00  
**Score Ceiling:** 0.95  
**Deadline:** 20 days

---

### Medium: India → Singapore (Multiple Compliance, No Tax ID)

**Scenario:** Manager with dependents relocating to Singapore.

**Required Actions:**
- **Documents:** Request and verify `passport`, `visa`, `employment_letter` (3 docs, **NOT** `work_permit`)
- **Departments:** HR + Legal approval
- **Compliance:** Register `payroll`, set `pdpa` consent, configure `shadow_payroll`
- **Critical:** Singapore does **NOT** require `tax_id` (calling it wastes steps)
- **Finalize:** Call `finalize_case` when all requirements met

**Difficulty Factors:**
- More compliance items than Easy (3 vs. 2)
- Two department approvals required
- Must recognize Singapore's unique requirements
- Optimal path: ~12 steps

**Expected Score Range:** 0.40 – 0.80  
**Score Ceiling:** 0.75  
**Deadline:** 25 days

---

### Hard: India → Germany + UAE (Multi-Country, Critical Trap)

**Scenario:** Director relocating simultaneously to Germany and UAE.

**Required Actions:**
- **Documents:** Request and verify `passport`, `visa`, `employment_letter`, `work_permit` (4 docs)
- **Departments:** HR + Legal + Finance approval
- **Compliance:** Register `tax_id` (Germany only) and `payroll`
- **Conflict Resolution:** Call `resolve_conflict` **before** `approve_finance`
- **Critical Trap:** **NEVER** call `set_tax_id:UAE` (UAE has no income tax, -0.25 penalty)
- **Finalize:** Call `finalize_case` when all requirements met

**Difficulty Factors:**
- Multi-country coordination
- Three department approvals (longest chain)
- Must resolve conflicts before Finance approval
- **UAE Tax Trap:** Pattern from Germany doesn't apply to UAE
- Optimal path: ~16 steps

**Expected Score Range:** 0.20 – 0.60  
**Score Ceiling:** 0.60  
**Deadline:** 30 days

---

## Setup and Usage Instructions

### Prerequisites
- Python 3.11+
- pip
- (Optional) Docker for containerized deployment

### Local Installation

```bash
# Clone the repository
git clone https://huggingface.co/spaces/Swayam14/openenv-workforce
cd openenv-workforce

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate.bat   # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables (for inference)
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_token_here"

# Start the API server
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Docker Deployment

```bash
# Build the image
docker build -t openenv-workforce .

# Run the container
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="your_token_here" \
  openenv-workforce
```

### Running Inference

```bash
# Configure environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_token_here"

# Run inference on all three tasks
python inference.py
```

### Running Tests

```bash
# Execute evaluation test suite (7 tests)
python test_eval.py
```

### API Usage Examples

**Start a new episode:**
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy"}'
```

**Take an action:**
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "action_type": "request_document",
    "target": "passport"
  }'
```

**Get current state:**
```bash
curl "http://localhost:7860/state?session_id=YOUR_SESSION_ID"
```

**Grade final state:**
```bash
curl -X POST http://localhost:7860/grade \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID"}'
```

---

## Baseline Scores

### Score Ranges by Task

| Task | Expected Range | Score Ceiling | Optimal Steps | Difficulty |
|------|----------------|---------------|---------------|------------|
| **Easy** (India → Germany) | 0.70 – 1.00 | 0.95 | ~11 | 🟢 Low |
| **Medium** (India → Singapore) | 0.40 – 0.80 | 0.75 | ~12 | 🟡 Medium |
| **Hard** (India → Germany + UAE) | 0.20 – 0.60 | 0.60 | ~16 | 🔴 High |

### Scoring Methodology

Scores are computed deterministically from state completeness using weighted components:

**Easy Task Weights (Total: 1.0):**
- Documents verified: 0.40 (0.10 per doc × 4)
- HR approved: 0.25
- Compliance complete: 0.25 (tax_id + payroll)
- Successfully finalized: 0.10

**Medium Task Weights (Total: 1.0):**
- Documents verified: 0.30 (0.10 per doc × 3)
- HR approved: 0.15
- Legal approved: 0.20
- Compliance complete: 0.25 (payroll + pdpa + shadow_payroll)
- Successfully finalized: 0.10

**Hard Task Weights (Total: 0.90, capped at 0.60):**
- Documents verified: 0.25 (0.0625 per doc × 4)
- HR approved: 0.08
- Legal approved: 0.08
- Finance approved: 0.08
- Compliance complete: 0.16 (tax_id + payroll)
- Conflict resolved: 0.15
- Successfully finalized: 0.10
- **UAE tax penalty:** -0.25 if `set_tax_id:UAE` called

### Penalties

**Parsimony Penalty:** -0.03 per unnecessary action (capped at -0.15)
- Example: Requesting `degree_certificate` when not required

**Rule Violation Penalty:** -0.25 to -0.30
- Example: Calling `set_tax_id:UAE` (UAE has no income tax)

**Prerequisite Penalty:** -0.20 per violation
- Example: Attempting Legal approval before documents verified

### Why Score Ceilings?

Ceilings ensure that perfect execution on harder tasks scores **lower** than partial execution on easier tasks, maintaining difficulty ordering:

```
Task      Ceiling   Perfect Agent Score
────────  ────────  ───────────────────
Easy      0.95      ~0.85–0.95
Medium    0.75      ~0.65–0.75
Hard      0.60      ~0.50–0.60
```

This design ensures that:
1. A flawless Easy run always scores higher than a flawless Hard run
2. Agents cannot "game" the system by cherry-picking hard tasks
3. Score ranges never overlap between difficulty tiers

### Sample Baseline Results

Running `python inference.py` with `gpt-4o-mini`:

```
EASY     | score=0.850 | steps=11 | success
MEDIUM   | score=0.680 | steps=13 | success
HARD     | score=0.420 | steps=17 | success

Average Score: 0.650
```

---

## License

This project is licensed under the MIT License.