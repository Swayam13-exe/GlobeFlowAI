# GlobeFlowAI: Teaching an Agent to Move People Across Borders

**An OpenEnv environment that compresses the messy reality of global mobility into a long-horizon, rule-rich, partially-observable RL task — and a benchmark that separates agents that have memorised a sequence from agents that have internalised a world model.**

By Team AI Kalesh · Built at the Meta PyTorch × OpenEnv Hackathon 2026, Scaler School of Technology

---

## The problem nobody puts in slide decks

When a multinational company relocates an engineer from Bangalore to Berlin, the Slack threads make it look like paperwork. The reality is that mobility teams are running a small distributed system with no error handling.

They are sequencing document submissions, chasing approvals across HR, Legal, and Finance, configuring host-country payroll, and resolving cases where two countries' rules genuinely contradict each other. When a regulation changes mid-process — and it does, regularly — they cannot restart the case from scratch. They have to detect the change, discard the now-invalid work, and re-route the remaining steps without dropping the employee.

This is not a toy workflow. Companies process thousands of these cases a year. The average successful relocation crosses six departments, twelve documents, and at least three jurisdictions when home, host, and tax-treaty country are counted separately. It is also exactly the kind of long-horizon, rule-rich, partially-observable task that exposes the difference between an agent that has memorised a sequence and one that has internalised the underlying world model.

We built GlobeFlowAI to make that distinction measurable.

---

## What the environment is built for

GlobeFlowAI is an OpenEnv-compatible reinforcement learning environment in which an agent handles the full lifecycle of an employee relocation case. The agent receives a typed observation describing the current case state, chooses one of twelve action types (with optional document and country targets), and receives a shaped per-step reward. Episodes terminate when the agent finalises the case, exhausts its step budget, or hits the deadline.

The environment exposes four tasks of increasing difficulty:

| Task       | Scenario                                     | Core competency                               |
| ---------- | -------------------------------------------- | --------------------------------------------- |
| **easy**   | Single engineer → Germany                    | Basic rule-following and action ordering      |
| **medium** | Manager with dependents → Singapore          | Country-specific rule differentiation         |
| **hard**   | Director → Germany + UAE simultaneously      | Multi-jurisdiction conflict resolution        |
| **crisis** | Germany relocation + mid-episode reg. change | Recovery from non-stationary rules            |

These are not difficulty-tagged variants of the same scenario. Each one tests a distinct competency that the prior task does not exercise. An agent that solves easy by memorising the action sequence will fail medium. An agent that passes medium will still fail hard and crisis for entirely different reasons. The four tasks form a diagnostic, not a leaderboard.

---

## The four tasks, told as a story

### Easy: a clean Germany relocation

The agent relocates an engineer to Germany. It must request and verify four documents (passport, visa, employment letter, work permit), obtain HR approval, register a tax ID, configure payroll, and finalise the case. Optimal play takes roughly eleven steps. A perfect episode scores ≈0.95 on the grader.

This is the warm-up. It teaches the agent the shape of the action space and the basic ordering of submit → verify → approve → configure → finalise.

### Medium: Singapore changes the rules

The agent relocates a manager with dependents to Singapore. The action space is identical to easy — the rules are not. Singapore does not require a tax ID; calling `set_tax_id` is a rule violation worth −0.30. Singapore does require PDPA consent and shadow payroll, which the easy task never used. Legal must also approve before finalisation, which easy did not require.

This is where rote memorisation breaks. An agent that learned the Germany sequence will confidently call `set_tax_id`, take the per-step penalty, and contaminate its grader score with a parsimony deduction. To pass medium, the agent must read the country field in the observation and select the action set that matches Singapore's rules — not the action set that matched Germany's.

### Hard: two countries, one employee, contradictory rules

The agent simultaneously relocates a director to both Germany and the UAE. Germany requires tax-ID registration. The UAE has no income tax. The case state ships with a pre-loaded `tax_conflict` that must be resolved with `resolve_conflict` before Finance will approve.

The trap is subtle. `set_tax_id` is the canonical right move for Germany, and the UAE is also in the country list. A pattern-matching agent will reach for `set_tax_id:UAE`, thinking it is being thorough. The environment penalises this twice — once at the per-step layer (−0.30) and once at the grader (−0.25 on the compliance component). A perfect episode scores ≈0.65, not because we capped it, but because the task genuinely has more weighted requirements to complete.

### Crisis: the rules change mid-case

The crisis task is the most distinctive piece of design in the environment.

The episode begins as a normal Germany relocation. The agent processes documents and approvals exactly as it would for easy. Then, at step 8, a regulatory event fires automatically: the Blue Card visa programme has been suspended, the existing visa document is marked invalid, and a new ICT-Permit document is injected into the case state. The observation surfaces a new top-priority action — `acknowledge_regulatory_change` — which the agent must call to clear the event before proceeding.

After acknowledging, the agent must request and verify the new `ict_permit` document while avoiding any further action targeting the invalidated visa.

This is non-stationary rules in a tightly controlled setting. We are not asking the agent to handle adversarial language or unstructured chaos. We are asking whether the agent will notice that its world model just changed — and re-plan accordingly. An agent that solved easy by memorising a fixed sequence will keep trying to verify the visa, collect the −0.30 penalty on each attempt, and finalise with a low grader score because it never acquired the ICT permit.

A perfect crisis episode scores ≈0.75. The crisis grader includes a partial-credit path for fired-but-unacknowledged events, because we wanted graceful failure to be distinguishable from total failure in the metrics.

---

## Why this design earns its complexity

Three deliberate design choices underlie everything.

**Shaped reward at every step, not at episode end.** Earlier environment designs gave reward only on finalisation — which is fine for large models with long context, and unworkable for small models trained with policy-gradient methods. GlobeFlowAI fires a progress-delta bonus at each step (capped at +0.20), milestone bonuses when entire requirement categories complete, and a per-step penalty for rule violations. This gives a small open-weights model usable gradient signal without requiring trainer-side reward shaping.

**No artificial grader ceilings.** An earlier version of the environment hard-capped per-task scores (0.949 for easy, 0.749 for medium, 0.599 for hard) to signal difficulty. We removed every cap. Difficulty now lives entirely in requirement weights — easy has 5 requirements summing to 0.95, hard has 8 summing to 0.65 — so a perfect agent on hard earns 0.65 because hard genuinely has more to do, not because we labelled it. The score is a clean function of agent behaviour.

**A parsimony penalty applies global pressure toward clean play.** Actions outside the task-relevant set cost −0.03 each, capped at −0.15 per episode. This stops agents from brute-forcing their way through the action space and gives the grader a signal that distinguishes "solved efficiently" from "solved eventually." It is also the quiet mechanism that makes trained and untrained agents separable in practice: untrained models tend to spam `request_document` when uncertain, and the penalty makes that legible.

---

## The result that changes the framing

*(This section will be completed with quantitative training results before the submission deadline. The design predictions below are empirically testable — we are documenting them in advance so readers can evaluate how well the training results bear them out.)*

We expect the difficulty-by-score curve to be **easy ≫ crisis ≥ medium ≫ hard**, not because hard is conceptually hardest in isolation but because hard demands the most independent rules to be respected simultaneously. Crisis is expected to show the largest gap between untrained and trained agents: pattern-matching solves easy by accident, but crisis requires the agent to actually read the observation at each step, which is a behaviour that only emerges with training pressure.

We also expect the parsimony penalty to be a meaningful signal in distinguishing trained from untrained agents, because untrained models tend to submit redundant document requests when uncertain.

If training inverts any of those predictions, that is itself a useful result — it tells us where our intuitions about the environment were wrong.

---

## Training the agent

We are training a reference agent with **GRPO + LoRA** on rollouts collected across all four tasks. The dense per-step reward and shaped progress bonus are intended to give a small model usable gradient signal.

> 🚧 **Training in active progress.** The Colab notebook, trained checkpoint, loss and reward curves, and the before/after evaluation table will be committed to the repository and embedded in this section before the submission deadline.

What will be reported once training completes:

- Base model and parameter count
- LoRA adapter configuration (rank, alpha, target modules)
- Optimiser, learning rate schedule, KL-divergence beta
- Training data composition across the four tasks
- Training loss curve and reward curve (embedded PNGs)
- Held-out evaluation: base model + 3-shot prompt vs. trained adapter, fresh seeds, all four tasks, crisis task as the load-bearing generalisation test

The rigorous A/B test that will separate RL contribution from prompt contribution: same prompt, same base model, same 3-shot examples — only the LoRA weights differ.

---

## What "non-stationary rules" actually measures

The crisis task is the benchmark's sharpest diagnostic. It measures something the other three tasks cannot: whether the agent maintains a live world model or replays a cached sequence.

The distinction matters for safety. An agent deployed on real mobility cases will encounter regulatory changes. The Blue Card suspension scenario is a controlled proxy for a real-world failure mode — a policy an agent relied on becomes invalid mid-task, and the agent must detect the change and update its plan rather than continue executing stale actions.

The structural check is strict: did the agent call `acknowledge_regulatory_change` before step 10? Did it subsequently request and verify `ict_permit`? Did it avoid further actions targeting the invalidated visa? The grader scores each of these as a separate weighted requirement. Partial credit for acknowledging-but-not-recovering is separated from credit for full recovery.

This mirrors the grounding insight in related work on agent oversight — getting the top-level verdict right is one skill, localising the evidence is another, and training is required to acquire the second one. Here, recognising a mid-episode misbehavior in the world (the regulatory change) is one skill; re-planning around it is another.

---

## The asymmetry between tasks

The four tasks reveal a pattern in failure modes that is worth flagging even before training results are available.

Medium and hard both penalise the same action — `set_tax_id` in a jurisdiction that has no income tax. But they surface it differently. In medium, the agent must simply not call `set_tax_id` at all (Singapore does not use it). In hard, the agent must call `set_tax_id` for Germany and not call it for UAE, in the same episode. An agent that learned "sometimes set_tax_id is wrong" will fail hard for the opposite reason from the agent that learned "set_tax_id is always right." The confusion matrix across the four tasks will surface which generalisation failure is more common — and that is the kind of finding you cannot surface without a benchmark that forces the distinction.

---

## Try it yourself

The environment is live on Hugging Face Spaces and accepts any HTTP client.

```bash
# Health check
curl https://huggingface.co/spaces/Swayam14/openenv-workforce/health

# Start a crisis episode
curl -X POST https://huggingface.co/spaces/Swayam14/openenv-workforce/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "crisis"}'

# Submit an action
curl -X POST https://huggingface.co/spaces/Swayam14/openenv-workforce/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "request_document", "target": "passport"}'
```

Running another agent on the leaderboard is one command once the eval script is published:

```bash
python scripts/eval.py \
    --model gpt-4o-mini --provider openai \
    --tasks easy medium hard crisis \
    --seeds 42 43 44 \
    --out results/eval_my_model.json
```

Drop the resulting JSON in a discussion thread on the model card and we will add the row.

Full quick-start, Docker setup, and the baseline OpenAI inference agent are in the [README](README.md).

---

## What's next

- **Richer crisis library.** The Blue Card suspension is one regulatory event. A generalising agent should handle visa category changes, work-permit quota suspensions, and bilateral tax-treaty renegotiations — each with different document implications.
- **LLM-driven adversarial doers** that generate hard-to-classify cases instead of the current deterministic rule policies.
- **Committee-based agent** — multiple sub-agents debate before submitting an action, with the environment scoring the committee's output.
- **Multi-step conflict resolution** where the conflict is distributed across several jurisdictions and requires sequential sub-actions to clear.
- **Cross-domain transfer** — train on mobility traces, evaluate on cross-border procurement and regulatory filings, which share the same workflow shape.
- **Parsimony penalty empirical study** — measure how the −0.03 per-step penalty interacts with base model size and training duration.

The contribution we are most invested in is not any single score on a leaderboard. It is that the environment makes the distinction between sequence-memorisation and world-model-internalisation measurable in a domain where that distinction carries real operational stakes.

---

Built at the Meta PyTorch × OpenEnv Hackathon 2026, Scaler School of Technology, by Team AI Kalesh.

MIT licensed.
