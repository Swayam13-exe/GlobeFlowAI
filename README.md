# GlobeFlowAI

Global Workforce Mobility & Compliance Orchestrator OpenEnv Environment

## Problem Description
GlobeFlowAI is an open environment simulating real-world employee relocation workflows. The agent is responsible for orchestrating multi-country compliance, processing essential documents, and receiving multiple corporate department approvals while detecting and avoiding potential rule violations across borders.

## System Design
The system uses a strict state machine to track relocation cases.
The environment logic is handled fully locally, with deterministic transitions and no external API reliance within the environment, ensuring speed and reproducibility. It is wrapped in a FastAPI server exposing standard OpenEnv JSON-RPC style endpoints (`/state`, `/step`, `/reset`).

## State + Actions
- **State**: A comprehensive JSON/Pydantic model storing case specifics, including employee details, document logs (missing, submitted, verified), compliance variables (payroll, tax, pdpa), and department approval statuses.
- **Actions**:
  - `request_document`
  - `verify_document`
  - `approve_hr`, `approve_legal`, `approve_finance`
  - `set_payroll`, `set_tax_id`, `set_shadow_payroll`, `set_pdpa`
  - `finalize_case`

## Tasks
1. **Easy**: `India -> Germany` - Basic Document validation routing.
2. **Medium**: `India -> Singapore` - Adding Payroll processing and strict Compliance mandates (PDPA).
3. **Hard**: `India -> Germany + UAE` - Multi-country conflict requiring intelligent bypass or case flag management, as Germany requires a tax footprint while UAE forbids it.

## Reward System
The environment operates with a dense reward signal designed for Reinforcement Learning or LLM Agents:
- `+0.2` for correct/valid transitions
- `+0.3` for progressing states (successfully verifying documents)
- `+0.5` for hitting core milestones (approvals)
- `+1.0` for finalizing successfully or correctly resolving a conflict
- Penalties exist for incorrect actions (`-0.2`), terminal rule violations (`-0.3`), and duplicate actions (`-0.1`).

## Setup
To run the server locally or in a Hugging Face Space:

```bash
docker build -t workforce .
docker run -p 7860:7860 workforce
```

Once running, the environment listens on `http://localhost:7860`.

## Baseline Results
- **Easy Task**: Baseline agents score averages of 1.0 reliably across 3-4 steps due to low branch complexity.
- **Medium Task**: Requires correct ordered evaluation (HR -> Legal -> Finance alongside Document verification before Legal).
- **Hard Task**: Requires recognizing conflicting boolean compliance requirements map against targeted countries and explicitly halting or routing around the violation. Baseline purely greedy LLM configurations commonly fail this mapping, highlighting reasoning deficits.

## Inference
An `inference.py` script is included. It uses an OpenAI-compatible client configuration to iterate the environment dynamically and automatically scores responses:
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-api-key"
python inference.py
```