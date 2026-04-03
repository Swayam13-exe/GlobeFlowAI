"""
inference.py
============
OpenAI-powered inference runner for openenv-workforce.

Runs an LLM agent through all three tasks (easy, medium, hard) by:
  1. Resetting the environment for the task
  2. Building a prompt with the current observation
  3. Calling the OpenAI API to get the next action
  4. Parsing and applying the action via env.step()
  5. Repeating until done=True or max_steps exceeded
  6. Grading the final state and printing a score table

Configuration (read from environment variables — NO hardcoding):
  API_BASE_URL:  OpenAI-compatible API base URL
                 Default: https://api.openai.com/v1
  MODEL_NAME:    Model to use  (e.g. gpt-4o, gpt-4-turbo)
                 Default: gpt-4o
  HF_TOKEN:      HuggingFace API token (used when running on HF Spaces)
                 Passed as Bearer token if set.
  OPENAI_API_KEY: Standard OpenAI key (used when HF_TOKEN not set)

Max steps per task: 20 (below the env ceiling of 25)

Author: Team AI Kalesh
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

import openai

# ---------------------------------------------------------------------------
# Environment variable configuration — NO hardcoding
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN:     str | None = os.environ.get("HF_TOKEN")
OPENAI_KEY:   str | None = os.environ.get("OPENAI_API_KEY")

# HF_TOKEN takes precedence over OPENAI_API_KEY
_api_key = HF_TOKEN or OPENAI_KEY or "MISSING_API_KEY"

MAX_STEPS_PER_TASK: int = 20   # hard ceiling for inference runner

# ---------------------------------------------------------------------------
# OpenAI client — configured once at module load
# ---------------------------------------------------------------------------

client = openai.OpenAI(
    api_key=_api_key,
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# System prompt — describes the action space to the LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert workforce mobility compliance agent.
Your goal is to resolve an employee relocation case by taking the correct sequence of actions.

Available action types and their targets:
  request_document   → passport | visa | employment_letter | degree_certificate
  verify_document    → passport | visa | employment_letter | degree_certificate
  approve_hr         → HR
  approve_legal      → Legal
  approve_finance    → Finance
  set_payroll        → Germany | Singapore | UAE
  set_tax_id         → Germany | Singapore  (⚠️ NEVER use UAE — UAE has NO income tax!)
  set_shadow_payroll → Singapore
  set_pdpa           → Singapore
  finalize_case      → all

CRITICAL RULES:
1. UAE has NO income tax. NEVER call set_tax_id with target=UAE.
2. Legal can only be approved AFTER all required documents are verified.
3. Finance can only be approved AFTER Legal is approved.
4. Singapore requires PDPA consent AND shadow payroll before finalize.
5. Only call actions that are shown in available_actions.

Always respond with ONLY a JSON object on a single line:
{"action_type": "<type>", "target": "<target>"}

No explanation. No markdown. Just the JSON."""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(observation: dict[str, Any]) -> str:
    """
    Build a concise user-facing prompt from the current observation.

    Args:
        observation: Observation dict from env reset/step.

    Returns:
        Prompt string for the LLM.
    """
    state      = observation.get("state", {})
    available  = observation.get("available_actions", [])
    blockers   = observation.get("current_blockers", [])
    last_res   = observation.get("last_action_result", "")
    last_err   = observation.get("last_action_error", "")
    steps      = observation.get("steps_taken", 0)

    progress   = state.get("progress", 0.0)
    deadline   = state.get("deadline_days", 0)
    countries  = state.get("countries", [])
    status     = state.get("status", "in_progress")

    # Document statuses
    docs_summary = []
    for doc_name, doc in state.get("documents", {}).items():
        docs_summary.append(f"  {doc_name}: {doc.get('status', 'unknown')}")

    # Department statuses
    depts = state.get("departments", {})
    dept_summary = []
    for dept, approved in depts.items():
        dept_summary.append(f"  {dept}: {'✓ approved' if approved else '✗ pending'}")

    # Compliance statuses
    comp = state.get("compliance", {})
    comp_summary = []
    for item, done in comp.items():
        comp_summary.append(f"  {item}: {'✓' if done else '✗'}")

    lines = [
        f"=== RELOCATION CASE ===",
        f"Countries: {', '.join(countries)}",
        f"Progress: {progress*100:.1f}%  |  Steps taken: {steps}  |  Deadline: {deadline} days",
        f"Status: {status}",
        "",
        "Documents:",
        *docs_summary,
        "",
        "Departments:",
        *dept_summary,
        "",
        "Compliance:",
        *comp_summary,
        "",
        f"Last action result: {last_res or 'n/a'}",
    ]

    if last_err:
        lines.append(f"Last error: {last_err}")

    if blockers:
        lines.append("")
        lines.append("Blockers remaining:")
        for b in blockers[:5]:   # show max 5 to keep prompt short
            lines.append(f"  - {b}")

    lines.append("")
    lines.append("Available actions (choose one):")
    for avail in available[:15]:  # cap at 15
        lines.append(f"  {avail}")

    lines.append("")
    lines.append("Respond with the single best action JSON:")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> dict[str, str] | None:
    """
    Extract JSON action from LLM response.

    Handles:
      - Clean JSON: {"action_type": "...", "target": "..."}
      - JSON inside markdown code fences
      - Trailing text after JSON

    Args:
        response_text: Raw string from the LLM.

    Returns:
        Dict with "action_type" and "target", or None if parsing fails.
    """
    # Try direct parse first
    text = response_text.strip()
    try:
        data = json.loads(text)
        if "action_type" in data and "target" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON object
    match = re.search(r'\{[^}]+\}', text)
    if match:
        try:
            data = json.loads(match.group())
            if "action_type" in data and "target" in data:
                return data
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(task_name: str) -> tuple[float, dict[str, Any]]:
    """
    Run one complete episode for the given task using the LLM agent.
    Emits structured [START], [STEP], and [END] logs to stdout.

    Args:
        task_name: "easy" | "medium" | "hard"

    Returns:
        (final_score: float, final_status: str)
    """
    from env.environment import WorkforceEnv
    from graders.graders import grade

    env = WorkforceEnv()
    observation = env.reset(task_name=task_name)

    obs_dict = observation.model_dump()

    # Emit [START] log
    print(json.dumps({
        "event": "START",
        "task": task_name,
        "countries": observation.state.countries,
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "max_steps": MAX_STEPS_PER_TASK,
    }))

    done     = False
    steps    = 0
    _captured_score:  float | None = None
    _captured_status: str   = "unknown"

    while not done and steps < MAX_STEPS_PER_TASK:
        prompt = build_prompt(obs_dict)

        # ── LLM call ────────────────────────────────────────────────────────
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,    # deterministic
                max_tokens=64,      # action JSON is tiny
            )
            raw_text = response.choices[0].message.content or ""
        except openai.OpenAIError as exc:
            print(json.dumps({
                "event": "STEP",
                "task": task_name,
                "step": steps + 1,
                "error": str(exc),
            }))
            break

        # ── Parse action ────────────────────────────────────────────────────
        action_dict = parse_action(raw_text)
        if not action_dict:
            print(json.dumps({
                "event": "STEP",
                "task": task_name,
                "step": steps + 1,
                "error": f"Failed to parse action from: {raw_text!r}",
            }))
            break

        from env.models import Action
        try:
            action = Action(**action_dict)
        except Exception as exc:
            print(json.dumps({
                "event": "STEP",
                "task": task_name,
                "step": steps + 1,
                "error": f"Invalid action structure: {exc}",
            }))
            break

        # ── Step ─────────────────────────────────────────────────────────────
        step_result = env.step(action)
        result_code = step_result.info.get("result", "?")
        reward      = step_result.reward
        done        = step_result.done

        # Emit [STEP] log
        print(json.dumps({
            "event": "STEP",
            "task": task_name,
            "step": steps + 1,
            "action_type": action.action_type,
            "target": action.target,
            "result": result_code,
            "reward": round(reward, 4),
            "done": done,
            "progress": round(obs_dict.get("state", {}).get("progress", 0.0), 4),
        }))

        obs_dict = step_result.observation.model_dump()
        steps   += 1

        # Capture score/status emitted by finalize_case for final reporting
        if step_result.done and "final_score" in step_result.info:
            _captured_score  = step_result.info["final_score"]
            _captured_status = step_result.info.get("status", "unknown")

        # Small delay to respect API rate limits
        time.sleep(0.1)

    # ── Final grade ──────────────────────────────────────────────────────────
    if _captured_score is not None:
        final_score = _captured_score
        final_status = _captured_status
    else:
        # Fallback: re-grade if finalize was never reached (max steps exceeded)
        final_state = env.state().model_dump()
        final_score = grade(task_name, final_state)
        final_status = final_state.get("status", "unknown")

    # Emit [END] log
    print(json.dumps({
        "event": "END",
        "task": task_name,
        "final_score": round(final_score, 4),
        "final_status": final_status,
        "steps_taken": steps,
    }))

    return final_score, final_status


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all three tasks and print a final score table."""
    from graders.graders import grade_all, print_score_table

    print(f"\nModel:    {MODEL_NAME}")
    print(f"API URL:  {API_BASE_URL}")
    print(f"Max steps per task: {MAX_STEPS_PER_TASK}")

    final_statuses: dict[str, str]            = {}
    scores:         dict[str, float]           = {}

    for task_name in ["easy", "medium", "hard"]:
        score, status = run_task(task_name)
        scores[task_name]       = score
        final_statuses[task_name] = status

    print_score_table(scores)


if __name__ == "__main__":
    main()