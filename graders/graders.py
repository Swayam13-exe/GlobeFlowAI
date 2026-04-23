"""
graders/graders.py
==================
Deterministic graders for all openenv-workforce tasks.

SCORING DESIGN:
  Each grader computes a score in (0.0, 1.0) from state completeness.
  Scores are based ONLY on what the task actually requires —
  consistent with required_departments and required_compliance in tasks.py.

  Task    Expected Range   What earns full score
  ──────  ──────────────   ─────────────────────
  easy    0.70 – 1.00      4 docs + HR + tax_id + payroll + finalized
  medium  0.40 – 0.80      3 docs + HR + Legal + payroll + pdpa + shadow + finalized
  hard    0.20 – 0.60      4 docs + HR + Legal + Finance + tax_id(DE) + payroll + no UAE tax + finalized
  crisis  0.40 – 0.90      4 docs (incl ict_permit) + HR + Legal + tax_id + payroll
                           + regulatory event acknowledged + finalized

Public API:
  grade(task_name, state) → float   main dispatcher used by environment.py
  explain(task_name, state) → str   human-readable breakdown for debugging

Author: Team AI Kalesh
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# CRITICAL: Epsilon to ensure scores are NEVER exactly 0.0 or 1.0
# ---------------------------------------------------------------------------

_EPSILON = 0.0001


# ---------------------------------------------------------------------------
# Documents per task — must match tasks.py exactly
# ---------------------------------------------------------------------------

_TASK_DOCS: dict[str, list[str]] = {
    "easy":   ["passport", "visa", "employment_letter", "work_permit"],
    "medium": ["passport", "visa", "employment_letter"],
    "hard":   ["passport", "visa", "employment_letter", "work_permit"],
    # crisis: ict_permit replaces visa after the event fires
    "crisis": ["passport", "employment_letter", "work_permit", "ict_permit"],
}

# ---------------------------------------------------------------------------
# Required compliance per task — must match tasks.py exactly
# ---------------------------------------------------------------------------

_TASK_COMPLIANCE: dict[str, list[str]] = {
    "easy":   ["tax_id", "payroll"],
    "medium": ["payroll", "pdpa", "shadow_payroll"],   # no tax_id for Singapore
    "hard":   ["tax_id", "payroll"],
    "crisis": ["tax_id", "payroll"],                   # Germany rules
}

# ---------------------------------------------------------------------------
# Required departments per task — must match tasks.py exactly
# ---------------------------------------------------------------------------

_TASK_DEPARTMENTS: dict[str, list[str]] = {
    "easy":   ["HR"],
    "medium": ["HR", "Legal"],
    "hard":   ["HR", "Legal", "Finance"],
    "crisis": ["HR", "Legal"],
}

# ---------------------------------------------------------------------------
# Task score ceilings
# ---------------------------------------------------------------------------

_TASK_CEILING: dict[str, float] = {
    "easy":   0.95,
    "medium": 0.75,
    "hard":   0.60,
    "crisis": 0.89,
}

# ---------------------------------------------------------------------------
# Valid actions per task — used for parsimony penalty
# ---------------------------------------------------------------------------

_EASY_VALID_ACTIONS: set[str] = {
    "request_document:passport",
    "verify_document:passport",
    "request_document:visa",
    "verify_document:visa",
    "request_document:employment_letter",
    "verify_document:employment_letter",
    "request_document:work_permit",
    "verify_document:work_permit",
    "approve_hr",
    "set_payroll",
    "set_tax_id",
    "finalize_case",
}

_MEDIUM_VALID_ACTIONS: set[str] = {
    "request_document:passport",
    "verify_document:passport",
    "request_document:visa",
    "verify_document:visa",
    "request_document:employment_letter",
    "verify_document:employment_letter",
    "approve_hr",
    "approve_legal",
    "set_payroll",
    "set_pdpa",
    "set_shadow_payroll",
    "finalize_case",
}

_HARD_VALID_ACTIONS: set[str] = {
    "request_document:passport",
    "verify_document:passport",
    "request_document:visa",
    "verify_document:visa",
    "request_document:employment_letter",
    "verify_document:employment_letter",
    "request_document:work_permit",
    "verify_document:work_permit",
    "approve_hr",
    "approve_legal",
    "approve_finance",
    "set_payroll",
    "set_tax_id",
    "resolve_conflict",
    "finalize_case",
}

_CRISIS_VALID_ACTIONS: set[str] = {
    "request_document:passport",
    "verify_document:passport",
    "request_document:visa",            # valid BEFORE event fires (steps 1-7)
    "verify_document:visa",             # valid BEFORE event fires (steps 1-7)
    "request_document:employment_letter",
    "verify_document:employment_letter",
    "request_document:work_permit",
    "verify_document:work_permit",
    "request_document:ict_permit",      # valid AFTER event fires
    "verify_document:ict_permit",       # valid AFTER event fires
    "acknowledge_regulatory_change",    # the key crisis action
    "approve_hr",
    "approve_legal",
    "set_payroll",
    "set_tax_id",
    "finalize_case",
}

_VALID_ACTIONS_MAP: dict[str, set[str]] = {
    "easy":   _EASY_VALID_ACTIONS,
    "medium": _MEDIUM_VALID_ACTIONS,
    "hard":   _HARD_VALID_ACTIONS,
    "crisis": _CRISIS_VALID_ACTIONS,
}


# ---------------------------------------------------------------------------
# Parsimony penalty
# ---------------------------------------------------------------------------

def _parsimony_penalty(prev_actions: list[str], valid_actions: set[str]) -> float:
    """
    -0.03 per action not in valid set. Capped at -0.15.
    System event markers ([SYSTEM_EVENT:...]) are never penalised.
    """
    junk = sum(
        1 for a in prev_actions
        if a not in valid_actions and not a.startswith("[SYSTEM_EVENT:")
    )
    return min(0.15, junk * 0.03)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def grade(task_name: str, state: dict[str, Any]) -> float:
    """
    Main entry point. Called by environment.py after every episode end.

    Args:
        task_name: "easy" | "medium" | "hard" | "crisis"
        state:     Episode state dict.

    Returns:
        Float strictly in (0.0, 1.0) — never exactly 0.0 or 1.0.
    """
    graders = {
        "easy":   grade_easy,
        "medium": grade_medium,
        "hard":   grade_hard,
        "crisis": grade_crisis,
    }
    if task_name not in graders:
        raise ValueError(
            f"Unknown task '{task_name}'. Valid: {list(graders.keys())}"
        )
    score = graders[task_name](state)

    if score <= 0.0:
        score = _EPSILON
    elif score >= 1.0:
        score = 1.0 - _EPSILON

    assert 0.0 < score < 1.0, f"Grader returned out-of-range score: {score}"
    return score


def explain(task_name: str, state: dict[str, Any]) -> str:
    """Return a human-readable scoring breakdown string."""
    reporters = {
        "easy":   _explain_easy,
        "medium": _explain_medium,
        "hard":   _explain_hard,
        "crisis": _explain_crisis,
    }
    if task_name not in reporters:
        raise ValueError(f"Unknown task '{task_name}'")
    return reporters[task_name](state)


# ---------------------------------------------------------------------------
# EASY grader — India → Germany
#
# Weights (sum = 1.0):
#   Documents verified    0.40   (4 docs, 0.10 each)
#   HR approved           0.25
#   Compliance            0.25   (tax_id 0.125 + payroll 0.125)
#   Finalized             0.10
# ---------------------------------------------------------------------------

def grade_easy(state: dict[str, Any]) -> float:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    status     = state.get("status", "in_progress")
    prev_acts  = state.get("previous_actions", [])

    required_docs = _TASK_DOCS["easy"]
    required_comp = _TASK_COMPLIANCE["easy"]

    # Documents (0.40)
    verified = sum(
        1 for d in required_docs
        if docs.get(d, {}).get("status") == "verified"
    )
    doc_score = (verified / len(required_docs)) * 0.40

    # HR approval (0.25)
    hr_score = 0.25 if depts.get("HR", False) else 0.0

    # Compliance (0.25)
    comp_done = sum(1 for c in required_comp if compliance.get(c, False))
    comp_score = (comp_done / len(required_comp)) * 0.25

    # Finalized (0.10)
    fin_score = 0.10 if status == "success" else 0.0

    raw = doc_score + hr_score + comp_score + fin_score

    # Parsimony penalty
    raw -= _parsimony_penalty(prev_acts, _EASY_VALID_ACTIONS)

    final_score = max(_EPSILON, min(0.9499, raw))
    final_score = round(final_score, 4)

    if final_score <= 0.0:
        final_score = _EPSILON
    elif final_score >= 1.0:
        final_score = 1.0 - _EPSILON

    return final_score


# ---------------------------------------------------------------------------
# MEDIUM grader — India → Singapore
#
# Weights (sum = 1.0):
#   Documents verified    0.30   (3 docs, 0.10 each)
#   HR approved           0.15
#   Legal approved        0.20
#   Compliance            0.25   (payroll 0.08 + pdpa 0.09 + shadow 0.08)
#   Finalized             0.10
# ---------------------------------------------------------------------------

def grade_medium(state: dict[str, Any]) -> float:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    status     = state.get("status", "in_progress")
    prev_acts  = state.get("previous_actions", [])

    required_docs = _TASK_DOCS["medium"]
    required_comp = _TASK_COMPLIANCE["medium"]

    # Documents (0.30)
    verified = sum(
        1 for d in required_docs
        if docs.get(d, {}).get("status") == "verified"
    )
    doc_score = (verified / len(required_docs)) * 0.30

    # HR (0.15)
    hr_score = 0.15 if depts.get("HR", False) else 0.0

    # Legal (0.20)
    legal_score = 0.20 if depts.get("Legal", False) else 0.0

    # Compliance (0.25)
    comp_done = sum(1 for c in required_comp if compliance.get(c, False))
    comp_score = (comp_done / len(required_comp)) * 0.25

    # Finalized (0.10)
    fin_score = 0.10 if status == "success" else 0.0

    raw = doc_score + hr_score + legal_score + comp_score + fin_score

    # Parsimony penalty
    raw -= _parsimony_penalty(prev_acts, _MEDIUM_VALID_ACTIONS)

    final_score = max(_EPSILON, min(0.7499, raw))
    final_score = round(final_score, 4)

    if final_score >= 0.75:
        final_score = 0.7499
    if final_score <= 0.0:
        final_score = _EPSILON

    return final_score


# ---------------------------------------------------------------------------
# HARD grader — India → Germany + UAE
#
# Weights (sum = 0.90, ceiling = 0.60):
#   Documents verified    0.25
#   HR approved           0.08
#   Legal approved        0.08
#   Finance approved      0.08
#   Compliance done       0.16  (tax_id 0.08 + payroll 0.08)
#   Conflict resolved     0.15
#   Finalized             0.10
#
# UAE trap penalty: -0.25 if set_tax_id:UAE called
# ---------------------------------------------------------------------------

def grade_hard(state: dict[str, Any]) -> float:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    conflicts  = state.get("conflicts", [])
    status     = state.get("status", "in_progress")
    prev_acts  = state.get("previous_actions", [])

    required_docs = _TASK_DOCS["hard"]
    required_comp = _TASK_COMPLIANCE["hard"]

    # Documents (0.25)
    verified = sum(
        1 for d in required_docs
        if docs.get(d, {}).get("status") == "verified"
    )
    doc_score = (verified / len(required_docs)) * 0.25 if required_docs else 0.0

    # HR (0.08)
    hr_score = 0.08 if depts.get("HR", False) else 0.0

    # Legal (0.08)
    legal_score = 0.08 if depts.get("Legal", False) else 0.0

    # Finance (0.08)
    finance_score = 0.08 if depts.get("Finance", False) else 0.0

    # Compliance (0.16 total)
    comp_done = sum(1 for c in required_comp if compliance.get(c, False))
    comp_score = (comp_done / len(required_comp)) * 0.16 if required_comp else 0.0

    # Conflict resolved (0.15)
    if conflicts:
        all_resolved = all(c.get("resolved", False) for c in conflicts)
        conflict_score = 0.15 if all_resolved else 0.0
    else:
        conflict_score = 0.0

    # Finalized (0.10)
    fin_score = 0.10 if status == "success" else 0.0

    raw = (
        doc_score + hr_score + legal_score + finance_score
        + comp_score + conflict_score + fin_score
    )

    # UAE tax violation penalty
    uae_tax_called = "set_tax_id:UAE" in prev_acts
    if uae_tax_called:
        raw -= 0.25

    # Parsimony penalty
    raw -= _parsimony_penalty(prev_acts, _HARD_VALID_ACTIONS)

    final_score = max(_EPSILON, min(0.5999, raw))
    final_score = round(final_score, 4)

    if final_score >= 0.60:
        final_score = 0.5999
    if final_score <= 0.0:
        final_score = _EPSILON

    return final_score


# ---------------------------------------------------------------------------
# CRISIS grader — India → Germany with mid-episode regulatory disruption
#
# Weights (sum = 1.0):
#   Docs verified (4)     0.35  (passport, employment_letter, work_permit, ict_permit)
#   HR approved           0.15
#   Legal approved        0.15
#   Compliance            0.15  (tax_id + payroll)
#   Regulatory handled    0.10  (event fired + acknowledged)
#   Finalized             0.10
#
# Penalty: using invalidated visa after event fires: -0.20 per occurrence
# Score ceiling: 0.89  |  Expected range: 0.40 – 0.90
#
# Key design intent:
#   Agent that ignores the event and finalizes with old visa scores < 0.40.
#   Agent that correctly adapts scores 0.75 – 0.89.
# ---------------------------------------------------------------------------

def grade_crisis(state: dict[str, Any]) -> float:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    status     = state.get("status", "in_progress")
    prev_acts  = state.get("previous_actions", [])

    # Required docs: ict_permit replaces visa
    required_docs = _TASK_DOCS["crisis"]
    required_comp = _TASK_COMPLIANCE["crisis"]

    # Documents (0.35)
    verified = sum(
        1 for d in required_docs
        if docs.get(d, {}).get("status") == "verified"
    )
    doc_score = (verified / len(required_docs)) * 0.35

    # HR (0.15)
    hr_score = 0.15 if depts.get("HR", False) else 0.0

    # Legal (0.15)
    legal_score = 0.15 if depts.get("Legal", False) else 0.0

    # Compliance: tax_id + payroll (0.15)
    comp_done = sum(1 for c in required_comp if compliance.get(c, False))
    comp_score = (comp_done / len(required_comp)) * 0.15

    # Regulatory event handling (0.10)
    event_fired = state.get("regulatory_event_fired", False)
    event_ack   = state.get("regulatory_event_acknowledged", False)
    if event_fired and event_ack:
        regulatory_score = 0.10
    elif event_fired and not event_ack:
        regulatory_score = 0.03   # partial — fired but not handled
    else:
        regulatory_score = 0.0    # hasn't fired yet

    # Finalized (0.10)
    fin_score = 0.10 if status == "success" else 0.0

    raw = doc_score + hr_score + legal_score + comp_score + regulatory_score + fin_score

    # Penalty: using invalidated visa AFTER the system event marker
    system_event_idx = next(
        (i for i, a in enumerate(prev_acts) if a.startswith("[SYSTEM_EVENT:")),
        None
    )
    if system_event_idx is not None:
        post_event_actions = prev_acts[system_event_idx:]
        visa_violations = sum(
            1 for a in post_event_actions
            if a in ("verify_document:visa", "request_document:visa")
        )
        raw -= visa_violations * 0.20

    # Parsimony penalty
    raw -= _parsimony_penalty(prev_acts, _CRISIS_VALID_ACTIONS)

    # Ceiling clamp
    final_score = max(_EPSILON, min(0.8899, raw))
    final_score = round(final_score, 4)

    if final_score >= 0.89:
        final_score = 0.8899
    if final_score <= 0.0:
        final_score = _EPSILON

    return final_score


# ---------------------------------------------------------------------------
# Explain helpers — human-readable breakdown
# ---------------------------------------------------------------------------

def _explain_easy(state: dict[str, Any]) -> str:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    status     = state.get("status", "in_progress")

    required_docs = _TASK_DOCS["easy"]
    required_comp = _TASK_COMPLIANCE["easy"]

    verified  = [d for d in required_docs if docs.get(d, {}).get("status") == "verified"]
    comp_done = [c for c in required_comp if compliance.get(c, False)]

    lines = [
        "=== EASY TASK SCORE BREAKDOWN ===",
        f"Documents:   {len(verified)}/{len(required_docs)} verified {verified}",
        f"HR:          {'✓' if depts.get('HR') else '✗'}",
        f"Compliance:  {len(comp_done)}/{len(required_comp)} done {comp_done}",
        f"Status:      {status}",
        f"Score:       {grade_easy(state):.4f}",
    ]
    return "\n".join(lines)


def _explain_medium(state: dict[str, Any]) -> str:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    status     = state.get("status", "in_progress")

    required_docs = _TASK_DOCS["medium"]
    required_comp = _TASK_COMPLIANCE["medium"]

    verified  = [d for d in required_docs if docs.get(d, {}).get("status") == "verified"]
    comp_done = [c for c in required_comp if compliance.get(c, False)]

    lines = [
        "=== MEDIUM TASK SCORE BREAKDOWN ===",
        f"Documents:   {len(verified)}/{len(required_docs)} verified {verified}",
        f"HR:          {'✓' if depts.get('HR') else '✗'}",
        f"Legal:       {'✓' if depts.get('Legal') else '✗'}",
        f"Compliance:  {len(comp_done)}/{len(required_comp)} done {comp_done}",
        f"Status:      {status}",
        f"Score:       {grade_medium(state):.4f}",
    ]
    return "\n".join(lines)


def _explain_hard(state: dict[str, Any]) -> str:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    conflicts  = state.get("conflicts", [])
    status     = state.get("status", "in_progress")
    prev_acts  = state.get("previous_actions", [])

    required_docs = _TASK_DOCS["hard"]
    required_comp = _TASK_COMPLIANCE["hard"]

    verified       = [d for d in required_docs if docs.get(d, {}).get("status") == "verified"]
    comp_done      = [c for c in required_comp if compliance.get(c, False)]
    uae_tax_called = "set_tax_id:UAE" in prev_acts
    resolved       = all(c.get("resolved", False) for c in conflicts) if conflicts else True

    lines = [
        "=== HARD TASK SCORE BREAKDOWN ===",
        f"Documents:        {len(verified)}/{len(required_docs)} verified {verified}",
        f"HR:               {'✓' if depts.get('HR') else '✗'}",
        f"Legal:            {'✓' if depts.get('Legal') else '✗'}",
        f"Finance:          {'✓' if depts.get('Finance') else '✗'}",
        f"Compliance:       {len(comp_done)}/{len(required_comp)} done {comp_done}",
        f"UAE no-tax rule:  {'✗ VIOLATED (-0.25)' if uae_tax_called else '✓ respected'}",
        f"Conflicts:        {'✓ resolved' if resolved else '✗ unresolved'}",
        f"Status:           {status}",
        f"Score:            {grade_hard(state):.4f}",
    ]
    return "\n".join(lines)


def _explain_crisis(state: dict[str, Any]) -> str:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    status     = state.get("status", "in_progress")
    prev_acts  = state.get("previous_actions", [])

    required_docs = _TASK_DOCS["crisis"]
    required_comp = _TASK_COMPLIANCE["crisis"]

    verified    = [d for d in required_docs if docs.get(d, {}).get("status") == "verified"]
    comp_done   = [c for c in required_comp if compliance.get(c, False)]
    event_fired = state.get("regulatory_event_fired", False)
    event_ack   = state.get("regulatory_event_acknowledged", False)

    system_event_idx = next(
        (i for i, a in enumerate(prev_acts) if a.startswith("[SYSTEM_EVENT:")),
        None
    )
    visa_violations = 0
    if system_event_idx is not None:
        post_event = prev_acts[system_event_idx:]
        visa_violations = sum(
            1 for a in post_event
            if a in ("verify_document:visa", "request_document:visa")
        )

    lines = [
        "=== CRISIS TASK SCORE BREAKDOWN ===",
        f"Documents:          {len(verified)}/{len(required_docs)} verified {verified}",
        f"  (requires ict_permit, NOT visa)",
        f"HR:                 {'✓' if depts.get('HR') else '✗'}",
        f"Legal:              {'✓' if depts.get('Legal') else '✗'}",
        f"Compliance:         {len(comp_done)}/{len(required_comp)} done {comp_done}",
        f"Event fired:        {'✓' if event_fired else '✗ (not yet)'}",
        f"Event acknowledged: {'✓' if event_ack else '✗ (-0.07 regulatory score)'}",
        f"Visa violations:    {visa_violations} post-event uses × -0.20",
        f"Status:             {status}",
        f"Score:              {grade_crisis(state):.4f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch utility used by inference.py
# ---------------------------------------------------------------------------

def grade_all(states: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Grade all tasks. Missing tasks default to epsilon floor."""
    results: dict[str, float] = {}
    for task_name in ["easy", "medium", "hard", "crisis"]:
        if task_name in states:
            try:
                results[task_name] = grade(task_name, states[task_name])
            except Exception as exc:
                results[task_name] = _EPSILON
                print(f"[grader] ERROR grading '{task_name}': {exc}")
        else:
            results[task_name] = _EPSILON
    return results