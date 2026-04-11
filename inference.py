"""
inference.py
============
OpenAI-powered inference runner for openenv-workforce.

MANDATORY STDOUT FORMAT — DO NOT MODIFY:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

MANDATORY ENV VARS:
  API_BASE_URL  — OpenAI-compatible API endpoint
  MODEL_NAME    — Model identifier
  HF_TOKEN      — API key (used as Bearer token)

Author: Team AI Kalesh
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment variables, NO hardcoding
# ---------------------------------------------------------------------------

API_BASE_URL: str        = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY:      str | None = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME:   str        = os.getenv("MODEL_NAME", "gpt-4o-mini")

BENCHMARK        = "openenv-workforce"
MAX_STEPS        = 20          # per task, below env ceiling of 25
TEMPERATURE      = 0.0         # deterministic
MAX_TOKENS       = 100
SUCCESS_THRESHOLD = 0.5        # score >= this → success=true in [END]

# ---------------------------------------------------------------------------
# Mandatory logging functions — exact format, do NOT change
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )



# ---------------------------------------------------------------------------
# System prompt for the LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert workforce mobility compliance agent.
Your goal is to resolve an employee relocation case by taking the correct actions.

At each step you receive the current case state.
Respond with ONLY a JSON object on a single line — no explanation, no markdown:
{"action_type": "<type>", "target": "<target>"}

Valid action_types and targets:
  request_document   → target = document name (passport/visa/employment_letter/work_permit)
  verify_document    → target = document name (must request first)
  approve_hr         → target = "" (no prerequisites)
  approve_legal      → target = "" (ALL documents must be verified first)
  approve_finance    → target = "" (Legal must approve first, conflicts resolved)
  set_payroll        → target = "" or country name
  set_tax_id         → target = "" or country name
  set_shadow_payroll → target = "" (Singapore only)
  set_pdpa           → target = "" (Singapore only)
  resolve_conflict   → target = "" (hard task only)
  finalize_case      → target = "" (only when all requirements met)

CRITICAL RULES:
1. ALWAYS request_document before verify_document
2. Legal approves ONLY after ALL documents are verified
3. Finance approves ONLY after Legal approves
4. UAE has NO income tax — NEVER call set_tax_id for UAE
5. Hard task: call resolve_conflict BEFORE approve_finance
6. Only call finalize_case when available_actions shows it
""").strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(obs: dict, step: int, history: List[str]) -> str:
    state     = obs.get("state", {})
    available = obs.get("available_actions", [])
    blockers  = obs.get("current_blockers", [])
    last_res  = obs.get("last_action_result", "")
    last_err  = obs.get("last_action_error", "")

    docs  = state.get("documents", {})
    depts = state.get("departments", {})
    comp  = state.get("compliance", {})

    doc_lines  = [f"  {k}: {v.get('status','?')}" for k, v in docs.items()]
    dept_lines = [f"  {k}: {'✓' if v else '✗'}" for k, v in depts.items()]
    comp_lines = [f"  {k}: {'✓' if v else '✗'}" for k, v in comp.items()]

    history_block = "\n".join(history[-4:]) if history else "None"

    lines = [
        f"Step: {step}",
        f"Countries: {state.get('countries', [])}",
        f"Progress: {state.get('progress', 0.0)*100:.1f}%  Deadline: {state.get('deadline_days', 0)} days",
        "",
        "Documents:",
        *doc_lines,
        "",
        "Departments:",
        *dept_lines,
        "",
        "Compliance:",
        *comp_lines,
        "",
        f"Last result: {last_res or 'n/a'}",
    ]

    if last_err:
        lines.append(f"Last error: {last_err}")

    conflicts = state.get("conflicts", [])
    if conflicts:
        lines.append("")
        for c in conflicts:
            resolved = c.get("resolved", False) if isinstance(c, dict) else False
            rule = c.get("rule", "") if isinstance(c, dict) else str(c)
            lines.append(f"Conflict: {rule} [{'RESOLVED' if resolved else 'UNRESOLVED'}]")

    if blockers:
        lines.append("")
        lines.append("Blockers:")
        for b in blockers[:4]:
            lines.append(f"  - {b}")

    lines.append("")
    lines.append("Available actions:")
    for a in available[:12]:
        lines.append(f"  {a}")

    lines.append("")
    lines.append(f"Recent history:\n{history_block}")
    lines.append("")
    lines.append("Respond with JSON only:")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def parse_action(text: str) -> dict:
    """Extract JSON action from LLM response. Returns safe fallback on failure."""
    text = text.strip()

    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                data = json.loads(part)
                if "action_type" in data:
                    return data
            except Exception:
                pass

    # Direct parse
    try:
        data = json.loads(text)
        if "action_type" in data:
            return data
    except Exception:
        pass

    # Extract first JSON object
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if "action_type" in data:
                return data
        except Exception:
            pass

    # Safe fallback
    return {"action_type": "request_document", "target": "passport"}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_model_action(
    client: OpenAI,
    obs: dict,
    step: int,
    history: List[str],
) -> dict:
    """Call the LLM and return parsed action dict."""
    prompt = build_prompt(obs, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_action(text)
    except Exception as exc:
        print(f"[DEBUG] Model call failed: {exc}", flush=True)
        return {"action_type": "request_document", "target": "passport"}


# ---------------------------------------------------------------------------
# Episode runner — one task
# ---------------------------------------------------------------------------

def run_episode(task_name: str, client: OpenAI) -> dict:
    """
    Run one complete episode for the given task.
    Emits [START], [STEP]×n, [END] to stdout.

    Returns summary dict for final table.
    """
    from env.environment import WorkforceEnv
    from graders.graders import grade

    rewards:      List[float] = []
    history:      List[str]   = []
    steps_taken:  int         = 0
    score:        float       = 0.0
    success:      bool        = False
    final_status: str         = "failed"

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = WorkforceEnv()

    try:
        obs = env.reset(task_name=task_name).model_dump()

        for step in range(1, MAX_STEPS + 1):
            # Check if already done from previous step
            if obs.get("done", False):
                break

            # Get action from model
            action_dict = get_model_action(client, obs, step, history)
            action_type = action_dict.get("action_type", "request_document")
            target      = action_dict.get("target", "")

            # Format action string for [STEP] log
            action_str = f"{action_type}:{target}" if target else action_type

            # Apply action
            try:
                from env.models import Action
                action = Action(action_type=action_type, target=target)
                result = env.step(action)

                reward = float(result.reward or 0.0)
                done   = bool(result.done)
                error  = result.info.get("error") if result.info else None

                # Update obs for next iteration
                obs = result.observation.model_dump()

                # Capture score if episode ended via finalize
                if done and "final_score" in result.info:
                    score        = float(result.info["final_score"])
                    final_status = result.info.get("status", "failed")

            except Exception as exc:
                reward = 0.0
                done   = False
                error  = str(exc)[:80]

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: {action_str} → reward={reward:+.2f}"
            )

            # Small delay to respect rate limits
            time.sleep(0.3)

            if done:
                break

        # If episode ended without finalize, grade the partial state
        if score == 0.0 or final_status == "failed":
            try:
                state_dict = env.state().model_dump()
                # Build grader-compatible dict
                score = grade(
                    task_name,
                    _flatten_state(state_dict),
                )
                final_status = state_dict.get("status", "failed")
            except Exception as exc:
                print(f"[DEBUG] Grading error: {exc}", flush=True)
                score = 0.0

        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        # [END] must always be emitted, even on exception
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return {
        "task":        task_name,
        "score":       score,
        "steps":       steps_taken,
        "status":      final_status,
        "cumulative":  round(sum(rewards), 4),
    }


# ---------------------------------------------------------------------------
# State flattener for grader compatibility
# ---------------------------------------------------------------------------

def _flatten_state(state_dict: dict) -> dict:
    """
    Convert WorkforceState.model_dump() output to the flat dict format
    that graders expect.
    """
    d = dict(state_dict)

    # Flatten documents
    docs = {}
    for name, doc in d.get("documents", {}).items():
        docs[name] = doc if isinstance(doc, dict) else {
            "status": doc.status, "is_valid": doc.is_valid
        }
    d["documents"] = docs

    # Flatten departments
    depts = d.get("departments", {})
    if not isinstance(depts, dict):
        depts = {"HR": depts.HR, "Legal": depts.Legal, "Finance": depts.Finance}
    d["departments"] = depts

    # Flatten compliance
    comp = d.get("compliance", {})
    if not isinstance(comp, dict):
        comp = {
            "tax_id": comp.tax_id, "payroll": comp.payroll,
            "pdpa": comp.pdpa, "shadow_payroll": comp.shadow_payroll,
        }
    d["compliance"] = comp

    # Flatten conflicts
    conflicts = d.get("conflicts", [])
    d["conflicts"] = [
        c if isinstance(c, dict) else {
            "countries": c.countries, "rule": c.rule, "resolved": c.resolved
        }
        for c in conflicts
    ]

    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    for task_name in ["easy", "medium", "hard"]:
        print(f"\n{'='*55}", flush=True)
        result = run_episode(task_name, client)
        results.append(result)

    # Final summary (plain text — won't interfere with [START]/[STEP]/[END] parsing)
    print(f"\n{'='*55}", flush=True)
    print("BASELINE RESULTS", flush=True)
    print(f"{'='*55}", flush=True)
    for r in results:
        print(
            f"{r['task'].upper():8} | score={r['score']:.4f} "
            f"| steps={r['steps']:2d} | {r['status']}",
            flush=True,
        )
    avg = sum(r["score"] for r in results) / len(results)
    print(f"\nAverage Score: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()