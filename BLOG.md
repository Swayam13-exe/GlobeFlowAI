# GlobeFlowAI: Teaching an Agent to Move People Across Borders

**An OpenEnv environment that compresses the messy reality of global mobility into a long-horizon, rule-rich, partially observable task — and a GRPO-trained Qwen2.5-1.5B agent that learns to handle it.**

*By Team AI Kalesh — Meta x Scaler OpenEnv AI Hackathon 2026*

---

## The problem nobody puts in slide decks

When a multinational company moves an engineer from Bangalore to Berlin, the Slack threads make it look like paperwork. The reality is that mobility teams are running a small distributed system with no error handling. They are sequencing document submissions, chasing department approvals across HR, Legal, and Finance, configuring host-country payroll, and resolving cases where two countries' rules genuinely contradict each other. When a regulation changes mid-process — and it does, regularly — they cannot restart the case from scratch. They have to detect the change, throw out the now-invalid work, and re-route the remaining steps without dropping the employee.

This is not a toy workflow. Companies deal with thousands of these cases a year. The average successful relocation crosses six departments, twelve documents, and at least three jurisdictions when home, host, and tax-treaty country are counted separately. It is also exactly the kind of long-horizon, rule-rich, partially observable task that exposes the difference between an agent that has memorised a sequence and one that has internalised the underlying world model.

We built GlobeFlowAI to make that distinction measurable.

---

## What GlobeFlowAI is

GlobeFlowAI is an OpenEnv-compatible reinforcement learning environment in which an agent handles the full lifecycle of an employee relocation case. The agent receives a typed observation describing the case state, chooses one of twelve action types (with optional targets for documents and countries), and receives a shaped reward at every step. Episodes terminate when the agent finalises the case, exhausts its step budget, or hits the deadline.

The environment exposes four tasks of increasing difficulty: a single-country relocation to Germany, a Singapore relocation with PDPA and shadow-payroll requirements, a multi-country Germany-plus-UAE case with a genuine tax-rule conflict, and a crisis task in which a regulatory event fires at step 8 and invalidates work the agent has already completed. The four tasks are not just difficulty-tagged variants of the same scenario — each one tests a distinct competence: rule-following, country-specific differentiation, conflict resolution, and recovery from non-stationary rules.

We were deliberate about not making this a clean toy environment. Real mobility workflows are messy because the rules genuinely contradict, and we wanted that mess in the design.

---

## The four tasks, told as a story

### Easy: a clean Germany relocation

The agent moves an engineer to Germany. It needs to request and verify four documents (passport, visa, employment letter, work permit), get HR approval, register a tax ID, configure payroll, and finalise the case. Optimal play takes about eleven steps. A perfect episode lands at approximately 0.95 on the grader. This is the warm-up — it teaches the agent the shape of the action space and the basic ordering of submit, verify, approve, configure, finalise.

### Medium: Singapore changes the rules

The agent moves a manager with dependents to Singapore. The action space is the same, but two rules quietly differ: Singapore does not require a tax ID (calling `set_tax_id` is a rule violation), and it does require PDPA consent and shadow payroll (which the easy task never used). Legal must also approve before finalisation, which the easy task did not need.

This is where rote memorisation breaks. An agent that has learned the easy-task sequence will confidently call `set_tax_id`, take the per-step penalty of -0.30, and contaminate its grader score with a parsimony deduction. To succeed on medium, the agent has to read the country field in the observation and pick the action set that matches Singapore's rules.

### Hard: two countries, one employee, contradictory rules

The agent simultaneously relocates a director to both Germany and the UAE. Germany requires tax-ID registration; the UAE has no income tax at all. The state ships with a pre-loaded `tax_conflict` that has to be cleared with `resolve_conflict` before Finance will approve.

The trap here is subtle and well-disguised. `set_tax_id` is the canonical right action for Germany, and the UAE is also in the country list, so a pattern-matching agent will reach for `set_tax_id:UAE` thinking it is being thorough. The environment punishes that twice — once at the per-step reward layer (-0.30) and once at the grader (-0.25). A perfect episode lands at approximately 0.65, lower than easy or medium because the same 100% completion now represents more requirements; the grader does not artificially cap the score, the workload genuinely is heavier.

### Crisis: the rules change mid-case

The crisis task is the most distinctive piece of design in the environment, and it is what we want judges to weight when they think about innovation.

The episode begins as a normal Germany relocation. The agent processes documents and approvals exactly as it would for the easy task. Then, at step 8, a regulatory event fires automatically: the Blue Card visa programme has been suspended, the existing visa document is marked invalid, and a new ICT-Permit document is injected into the state. The observation surfaces this through a new top-priority action — `acknowledge_regulatory_change` — which the agent must call to clear the event. The agent must then request and verify the new `ict_permit` document, while avoiding any further action targeting the now-invalidated visa.

This is non-stationary rules in a tightly controlled setting. We are not asking the agent to handle adversarial language or unstructured chaos; we are asking whether it will notice that its world model just changed and re-plan accordingly. An agent that solved the easy task by memorising a fixed sequence will fail this one — it will keep trying to verify the visa, take the -0.30 penalty repeatedly, and finalise with a low grader score because it never picked up the ICT permit.

A perfect crisis episode lands at approximately 0.75. The crisis grader is the only one that has a partial-credit path for fired-but-unacknowledged events, because we wanted to see graceful failure separated from total failure in the metrics.

---

## Why this design earns its complexity

The environment makes three deliberate design choices that we think judges should weigh under the *Environment Innovation* and *Reward Design* criteria.

**The reward function is shaped at every step, not at episode end.** Documents, departments, compliance, and conflict resolution each contribute weighted progress, and the per-step reward includes a progress-delta bonus capped at +0.20. This gives a small open-weights model usable gradient signal without forcing the trainer to add reward shaping of its own. The reward also fires milestone bonuses when entire categories complete, which front-loads value into the early phase of an episode and makes partial completions distinguishable from total failure.

**The grader has no artificial ceilings.** Earlier iterations of the environment used per-task ceilings (0.949 for easy, 0.749 for medium, 0.599 for hard) to encode difficulty. We removed them entirely. Difficulty now lives in the requirement weights — easy has 5 weighted requirements summing to 0.95, hard has 8 summing to 0.65 — so the score is a clean function of agent behaviour, not a labelling artefact. A perfect agent on a hard task earns 0.65 because the task genuinely is harder, not because we hand-capped it.

**A parsimony penalty applies global pressure toward clean play.** Actions outside the task-relevant set incur -0.03 each, capped at -0.15 per episode. This stops agents from spamming the action space to brute-force their way through, and gives the grader a signal that distinguishes "solved efficiently" from "solved eventually." It is the reward design's quiet contribution to making the four tasks separable.

We also red-teamed our own environment during development. The grader has an `assert 0.0 < score < 1.0` that catches any out-of-range score before it leaves the function. The action history records only completed and rule-violating actions, so an agent can legitimately retry an action that previously failed because a prerequisite was not yet met — which means the environment does not punish exploration. The crisis event-firing logic is keyed on `_steps_taken`, which only increments after a successful step, so penalty steps do not advance the crisis clock and an agent cannot accidentally skip the event by failing fast.

---

## Training the agent

We trained a Qwen2.5-1.5B-Instruct policy with **GRPO (Group Relative Policy Optimization)** through Hugging Face TRL, with a LoRA adapter applied to the attention projection matrices. The full pipeline runs end-to-end in approximately 12 minutes on a single T4 GPU — accessible enough that any judge can reproduce it on a free Colab instance.

### Pipeline at a glance

| Component | Choice |
|-----------|--------|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Method | GRPO via TRL, LoRA via PEFT |
| LoRA configuration | rank 16, alpha 32, dropout 0.05; targets q/k/v/o projections |
| Learning rate | 1e-4 |
| KL beta | 0.001 |
| Generations per step | 4 |
| Epochs | 2 |
| Time budget | 12 minutes (hard stop) |

The training reward is what makes this pipeline work for a small base model. We do not ask the trainer to bring its own reward shaping; we hand the GRPO loop the live environment and let it compute rewards from real rollouts. For each completion the trainer parses the JSON action, resets the environment, replays a small prefix of actions, and takes the proposed step. The reward is the real per-step reward plus a 2x progress-delta bonus plus the final score if the episode terminates within the rollout, all clipped to [-0.5, 0.5]. Completions that fail to parse are penalised at -0.3, and actions outside `available_actions` are penalised at -0.5. This is what keeps a 1.5-billion-parameter model on-distribution for a structured action space it has never seen before.

The training dataset is small by design: 16 prompts spanning all four tasks. Half are taken at episode start so the policy learns the opening moves; half pre-advance the environment by 8 steps so the policy also trains on near-completion states and learns when to call `finalize_case`. This is the single change that converted earlier runs (where the model could open episodes but never finalise them) into a pipeline that produces measurable lifts on every task.

### What the training looks like

![GlobeFlowAI training results](../assets/training_results.png)

The training reward climbs from -0.05 at step 0 to approximately +0.48 by step 14, with the smoothed trace showing a clear inflection after step 6 — that is the point at which the policy stops invalid-formatting and starts producing well-formed actions that the environment actually executes. The loss curve oscillates near zero with a handful of larger steps, which is exactly the behaviour we expect from a policy-gradient method at this scale and at this level of completion variance. Both signals are healthy.

### Before vs after

Evaluation uses temperature 0.3 and `enforce_available=True`, with a single rollout per task. The before column is the same Qwen2.5-1.5B-Instruct model under identical prompting, with no adapter loaded.

| Task | Before | After | Absolute lift | Relative lift |
|------|--------|-------|---------------|---------------|
| Easy | 0.750 | 0.800 | +0.050 | +6.7% |
| Medium | 0.590 | 0.660 | +0.070 | +11.9% |
| Hard | 0.450 | 0.540 | +0.090 | +20.0% |
| Crisis | 0.550 | 0.660 | +0.110 | +20.0% |
| **Average** | **0.585** | **0.665** | **+0.080** | **+13.7%** |

The shape of this table is the headline finding. The easy task — the one where pattern matching alone gets you most of the way — improves the least, and the crisis task — the one where the agent has to actually adapt to a non-stationary rule change — improves the most. Both hard and crisis improve by 20% relative. This is consistent with the design prediction: the environment was built so that shaped reward and dense progress signal would give a small model usable gradient on tasks where surface patterns fail, and the relative-lift column confirms that prediction empirically.

It is also worth noting that the crisis task's after-training score (0.660) is materially closer to its perfect score (approximately 0.75) than the hard task's (0.540 vs approximately 0.65). After training, the agent is operating at roughly 88% of its ceiling on crisis and 83% on hard. The agent benefited more from training pressure on long-horizon adaptation than on multi-rule conflict resolution — a finding we did not predict in advance, and one that suggests crisis-style tasks may be a particularly fertile direction for environment design more broadly.

---

## Limitations we are honest about

The training run is intentionally tight on a free-tier T4: 12 minutes, 16 prompts, two epochs. This is what makes it reproducible on Colab. It is not what makes it state-of-the-art. A larger base model, longer training, and multi-seed evaluation with variance bars would all sharpen the numbers. We chose to ship a pipeline that any judge can reproduce in a coffee break, and we treat the table above as a lower bound on what the environment can support, not an upper bound on what the policy can learn.

Single-rollout evaluation per task is the most defensible target for criticism, and we agree with that criticism. Multi-seed evaluation is the obvious next step.

---

## Try it yourself

The environment is live on Hugging Face Spaces and can be hit from any HTTP client. The full source is on GitHub.

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

The full quick-start, including local installation, Docker, and the baseline OpenAI inference agent, is in the [README](https://github.com/Swayam14/openenv-workforce#setup-and-usage).

---

## What's next

The environment as it stands tests four well-separated competences. The natural next steps, in priority order, are: a richer crisis library so that the regulatory-disruption mechanic generalises beyond the single Blue Card scenario; multi-seed evaluation across all four tasks so that the lifts above carry confidence intervals; and a port of the multi-country conflict-resolution pattern to adjacent enterprise domains, because the same workflow shape shows up in cross-border procurement, regulatory filings, and IP transfers.

We also think the parsimony penalty deserves more empirical work. It is a small piece of the reward design, but it is the piece that makes the difference between "solved" and "solved cleanly," and we suspect it interacts in interesting ways with the choice of base model size.

---

## Acknowledgements

GlobeFlowAI was built at the **Meta x Scaler OpenEnv AI Hackathon 2026** by **Team AI Kalesh**. We want to thank the Meta PyTorch and Scaler School of Technology teams for designing a hackathon that takes environment design seriously — environments are infrastructure, and infrastructure deserves first-class attention. We also want to acknowledge the OpenEnv specification itself, which made it possible for us to spend our time thinking about workflow design rather than HTTP plumbing.

The full source, the live Hugging Face Space, the training notebook, and this blog post are all linked from the [README](https://github.com/Swayam14/openenv-workforce).