---
title: GlobeFlowAI
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# GlobeFlowAI

**An OpenEnv environment for training and measuring AI agents that manage complex, rule-rich, multi-jurisdiction employee relocation workflows.**

When a multinational company relocates an engineer across borders, the process crosses six departments, twelve documents, and at least three jurisdictions — and the rules genuinely contradict each other. GlobeFlowAI compresses that reality into a long-horizon, partially-observable reinforcement learning environment. The agent must sequence document submissions, obtain cross-department approvals, configure host-country payroll, and recover mid-episode when regulations change.

**Design principle:** four tasks that test four distinct competencies — rule-following, country-specific differentiation, conflict resolution, and recovery from non-stationary rules. Rote memorisation solves the easy task and fails the rest.

## Submission

| What                          | Link                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------- |
| Hugging Face Space (live env) | https://huggingface.co/spaces/Swayam14/openenv-workforce                                          |
| Colab notebook (training)     | *(forthcoming — will be committed before submission deadline)*                                    |
| Code repository               | https://github.com/Swayam14/openenv-workforce                                                     |
| Blog / writeup                | [BLOG.md](BLOG.md)                                                                                |
| Trained model adapter         | *(forthcoming — will be committed before submission deadline)*                                    |

---

🔗 **Live HF Space:** https://huggingface.co/spaces/Swayam14/openenv-workforce

🔗 **Source:** https://github.com/Swayam14/openenv-workforce

📓 **Colab training notebook:** *(forthcoming)*

📝 **Writeup:** [BLOG.md](BLOG.md)

👥 **Team:** AI Kalesh

---

## What this env measures

An agent receives a typed observation describing the current relocation case state, selects one of twelve action types (with optional document and country targets), and receives a shaped per-step reward. Episodes end on finalisation, step-budget exhaustion, or deadline breach.

| Task       | Scenario                                         | Key competency tested                         |
| ---------- | ------------------------------------------------ | --------------------------------------------- |
| **easy**   | Single engineer → Germany                        | Basic action ordering and rule-following       |
| **medium** | Manager with dependents → Singapore              | Country-specific differentiation               |
| **hard**   | Director → Germany + UAE simultaneously          | Multi-jurisdiction conflict resolution         |
| **crisis** | Germany relocation with mid-episode reg. change  | Recovery from non-stationary rules             |

Task names encode **generalization difficulty**, not per-episode difficulty. A random policy scores well below chance on all four tasks — the difficulty curve only emerges once the agent actually trains.

---

## Why this env is different

**Shaped per-step reward, not just episode-end signal.** Documents, departments, compliance milestones, and conflict resolution each contribute weighted progress. A progress-delta bonus (capped at +0.20 per step) and categorical milestone bonuses give a small open-weights model usable gradient signal without trainer-side reward shaping.

**No artificial grader ceilings.** Earlier iterations capped per-task scores (0.949 / 0.749 / 0.599) to encode difficulty. Those ceilings are removed. Difficulty now lives in requirement weights — easy has 5 requirements summing to 0.95, hard has 8 summing to 0.65. Score is a clean function of agent behaviour.

**Non-stationary rules, tightly controlled.** In the crisis task, a regulatory event fires at step 8: the Blue Card visa is suspended, and an ICT Permit is injected into the state. The agent must acknowledge the change and re-plan accordingly. An agent that memorised the easy-task sequence will keep verifying the invalidated visa, collect the −0.30 penalty per step, and finalise with an incomplete case.

**Parsimony penalty for clean play.** Actions outside the task-relevant set incur −0.03 each, capped at −0.15 per episode. This distinguishes "solved efficiently" from "solved eventually" — a signal that separates trained from untrained agents in practice.

**Gaming-hardened reward.** The grader has an `assert 0.0 < score < 1.0` guard on every exit path. Action history records only completed and rule-violating actions, so the agent can legitimately retry a failed action after resolving its prerequisite — exploration is not punished. The crisis clock advances only on successful steps, so penalty steps cannot accidentally skip the regulatory event.

**Model-agnostic by API.** The `/step` endpoint accepts any raw action dict. Swap in a hand-tuned prompt, a GRPO-trained LoRA, a chain-of-thought planner, or a rule-based heuristic — the environment scores them all identically.

---

## Training the agent

We are training a reference overseer with **GRPO + LoRA** on rollouts collected across all four tasks. The dense per-step reward and shaped progress bonus are designed to give a small model usable gradient signal without requiring extensive trainer-side modifications.

> 🚧 **Training in active progress.** The training script, hyperparameters, loss and reward curves, and before/after benchmark numbers will be committed to the repository and linked here before the submission deadline.

Once training completes, this section will document:

- Base model and parameter count
- LoRA adapter configuration (rank, alpha, target modules)
- Optimiser, LR schedule, and KL-divergence beta
- Training data composition across the four tasks
- Loss curve and reward curve (embedded inline as PNGs)
- Held-out evaluation methodology — fresh seeds across all four tasks, crisis task as the load-bearing generalisation test
- Before/after comparison: base model with 3-shot prompting vs. trained adapter, same evaluation set

---

## Out-of-distribution evaluation

The point of the env is to distinguish an agent that has internalised the underlying world model from one that has memorised a sequence. The four tasks are not variants of a single scenario — each tests a distinct competency that the previous task does not exercise.

Expected results once training completes:

| Condition                          | Easy (target) | Medium (target) | Hard (target) | Crisis (target) |
| ---------------------------------- | ------------- | --------------- | ------------- | --------------- |
| Random policy                      | ≪ 0.60        | ≪ 0.55          | ≪ 0.45        | ≪ 0.50          |
| Base model + 3-shot prompt         | TBD           | TBD             | TBD           | TBD             |
| **GRPO-LoRA (ours)**               | **TBD**       | **TBD**         | **TBD**       | **TBD**         |

Raw eval JSONs will be committed to `results/` once training is complete.

---

## Quick start

### Hit the live env

```bash
# Health check
curl https://huggingface.co/spaces/Swayam14/openenv-workforce/health

# List tasks
curl https://huggingface.co/spaces/Swayam14/openenv-workforce/tasks

# Start a crisis episode
curl -X POST https://huggingface.co/spaces/Swayam14/openenv-workforce/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "crisis"}'

# Submit an action
curl -X POST https://huggingface.co/spaces/Swayam14/openenv-workforce/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "request_document", "target": "passport"}'

# Get the random baseline
curl https://huggingface.co/spaces/Swayam14/openenv-workforce/baseline
```

### Run locally

```bash
git clone https://github.com/Swayam14/openenv-workforce.git
cd openenv-workforce
pip install -r requirements.txt
pip install -e .
python -m server.app          # serves on :7860
pytest tests/ -q              # all tests should pass
```

### Reproduce the eval *(once checkpoint is published)*

```bash
# Pull trained adapter from HF
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download(repo_id='<team>/globeflowai-adapter', \
  local_dir='checkpoints/final')"

# Run eval across all four tasks (GPU recommended)
python scripts/eval.py \
  --model-path checkpoints/final \
  --tasks easy medium hard crisis \
  --seeds 42 43 44 \
  --out results/repro.json
```

---

## Tasks

| Task       | Training distribution      | Success criteria                              |
| ---------- | -------------------------- | --------------------------------------------- |
| `easy`     | Germany relocation         | Episode score > 0.90                          |
| `medium`   | Singapore relocation       | Episode score > 0.75, no rule violations      |
| `hard`     | Germany + UAE relocation   | Episode score > 0.55, conflict resolved       |
| `crisis`   | Germany + mid-ep reg event | Event acknowledged, ICT permit acquired       |

Perfect play expected scores: easy ≈ 0.95 · medium ≈ 0.85 · hard ≈ 0.65 · crisis ≈ 0.75. Scores are lower on harder tasks because those tasks genuinely have more weighted requirements — there is no artificial cap.

---

## Action schema

The action is a JSON dict submitted to `/step`:

```json
{
  "action_type": "<see table below>",
  "target": "<document name or country code, if applicable>"
}
```

| `action_type`                | Description                                              |
| ---------------------------- | -------------------------------------------------------- |
| `request_document`           | Initiates a document request (target = document name)    |
| `verify_document`            | Marks a previously requested document as verified        |
| `get_hr_approval`            | Requests HR department approval                          |
| `get_legal_approval`         | Requests Legal department approval                       |
| `get_finance_approval`       | Requests Finance department approval                     |
| `set_tax_id`                 | Registers a tax ID (target = country; rule-restricted)   |
| `configure_payroll`          | Configures host-country payroll                          |
| `resolve_conflict`           | Resolves a loaded conflict in the case state             |
| `acknowledge_regulatory_change` | Clears a fired regulatory event (crisis task only)    |
| `request_pdpa_consent`       | Requests PDPA data-protection consent (Singapore only)   |
| `configure_shadow_payroll`   | Configures shadow payroll (Singapore only)               |
| `finalise_case`              | Terminates the episode and triggers grader scoring       |

Invalid actions (wrong target type, action not applicable to current state) return a structured error; the episode continues.

---

## Reward function

### Per-step reward

| Event                                        | Reward             |
| -------------------------------------------- | ------------------ |
| Document verified (weighted by task)         | +0.05 to +0.15     |
| Department approval obtained                 | +0.10              |
| Category milestone completed                 | +0.15 bonus        |
| Progress delta (vs. prior step, capped)      | up to +0.20        |
| Rule violation (e.g., `set_tax_id` in UAE)   | −0.30              |
| Redundant / out-of-scope action              | −0.03 (cap −0.15)  |

### Episode-end grader score

| Component                          | Weight   |
| ---------------------------------- | -------- |
| Document completion                | 0.30     |
| Departmental approvals             | 0.25     |
| Compliance requirements            | 0.25     |
| Conflict resolution (hard/crisis)  | 0.20     |

Grader score is clamped to (0.0, 1.0). Every constant-action strategy scores below a minimally-informed random policy on the full task suite.

---

## Empirical sanity checks

**Random baseline (uniform action sampling, n=20 per task):**

| Task    | Mean episode score | Rule violations / ep |
| ------- | ------------------ | -------------------- |
| easy    | < 0.25             | ~3                   |
| medium  | < 0.20             | ~5                   |
| hard    | < 0.15             | ~7                   |
| crisis  | < 0.20             | ~4                   |

Random is well below chance on all tasks.

**Adversarial robustness:** malformed JSON, unknown `action_type` values, missing `target` fields, oversized payloads, negative seeds, and concurrent `/reset` calls have all been probed. The server returns structured 4xx errors, never 500s. No stack-trace leaks.

**Determinism:** `reset(seed=N)` twice returns the same initial observation and the same crisis-event timing. Confirmed across all four tasks and multiple seeds.

**Crisis clock integrity:** verified that the crisis event fires on `_steps_taken == 8`, not on wall-clock time or total API calls. Penalty steps do not advance the counter.

---

## Architecture

```
[ Relocation case state ]  ──►  [ Agent ]
  (partially-observable,           (any LLM or policy,
   rule-loaded, multi-country)      submits JSON action dict)
                                          │
                                          ▼
                                   action_type + target
                                          │
                                          ▼
                              [ Per-step reward + next obs ]
                                          │
                             (on finalise_case or budget exhaustion)
                                          │
                                          ▼
                                  [ Grader score ]
                              (weighted requirement completion)
```

The case state is deterministic Python — ground-truth rule labels are always known. This is deliberate: controlled rule-sets are the only way to measure whether an agent actually learns the rules vs. memorises a sequence. Any LLM or policy can plug in as the agent.

---

## Limitations we report honestly

- **Training is incomplete at submission time.** The reference LoRA numbers are not yet available. This section will be updated with full results, curves, and checkpoint links before the deadline.
- **Doers are deterministic Python policies, not LLM-driven.** This is a controlled lab. Extending to LLM-generated adversarial cases (e.g., ambiguous observations, contradictory document states) is a planned extension.
- **Crisis library has one scenario.** The Blue Card suspension is the only regulatory event in v1. Generalisation across multiple distinct crisis types is future work.
- **Parsimony penalty interaction with base model size is empirically unexplored.** We expect it to be a meaningful signal separating trained from untrained agents, but have not yet measured this across model scales.

---

## Future work

- Richer crisis library — multiple distinct regulatory event types beyond Blue Card
- LLM-in-the-loop adversarial case generation
- Committee-based agent (multiple sub-agents debate before submitting action)
- Multi-step conflict resolution (conflict distributed across several jurisdictions)
- Port workflow pattern to adjacent domains: cross-border procurement, regulatory filings, IP transfers
- Empirical study of parsimony penalty interaction with base model size

---

## Citation

If you use GlobeFlowAI in research, please cite:

```
@software{globeflowai2026,
  title  = {GlobeFlowAI: An OpenEnv environment for AI-driven global mobility workflows},
  author = {Team AI Kalesh},
  year   = {2026},
  url    = {https://github.com/Swayam14/openenv-workforce}
}
```

Built at the Meta PyTorch × OpenEnv Hackathon 2026, Scaler School of Technology.

---

## License

This project is licensed under the MIT License.