"""
graders/graders.py
==================
Deterministic graders for all three openenv-workforce tasks.

SCORING DESIGN:
  Each grader computes a score in [0.0, 1.0] from state completeness.
  Scores are based ONLY on what the task actually requires —
  consistent with required_departments and required_compliance in tasks.py.

  Task    Expected Range   What earns full score
  ──────  ──────────────   ─────────────────────
  easy    0.70 – 1.00      4 docs + HR + tax_id + payroll + finalized
  medium  0.40 – 0.80      3 docs + HR + Legal + payroll + pdpa + shadow + finalized
  hard    0.20 – 0.60      4 docs + HR + Legal + Finance + tax_id(DE) + payroll + no UAE tax + finalized

Public API:
  grade(task_name, state) → float   main dispatcher used by environment.py
  explain(task_name, state) → str   human-readable breakdown for debugging

Author: Team AI Kalesh
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Documents per task — must match tasks.py exactly
# ---------------------------------------------------------------------------

# FIX 1 + FIX 6: Germany uses work_permit not degree_certificate
_TASK_DOCS: dict[str, list[str]] = {
    "easy":   ["passport", "visa", "employment_letter", "work_permit"],
    "medium": ["passport", "visa", "employment_letter"],
    "hard":   ["passport", "visa", "employment_letter", "work_permit"],
}

# ---------------------------------------------------------------------------
# Required compliance per task — must match tasks.py exactly
# ---------------------------------------------------------------------------

# FIX 2: Singapore does NOT require tax_id
_TASK_COMPLIANCE: dict[str, list[str]] = {
    "easy":   ["tax_id", "payroll"],
    "medium": ["payroll", "pdpa", "shadow_payroll"],   # no tax_id for Singapore
    "hard":   ["tax_id", "payroll"],                   # Germany tax_id + payroll
}

# ---------------------------------------------------------------------------
# Required departments per task — must match tasks.py exactly
# ---------------------------------------------------------------------------

_TASK_DEPARTMENTS: dict[str, list[str]] = {
    "easy":   ["HR"],               # FIX 3: easy only needs HR
    "medium": ["HR", "Legal"],      # FIX 4: medium needs HR + Legal only
    "hard":   ["HR", "Legal", "Finance"],
}

# ---------------------------------------------------------------------------
# Task score ceilings — enforces difficulty ordering easy > medium > hard
# A perfect agent scores AT MOST this value, ensuring:
#   easy   ~0.78-0.95  (straightforward, few steps)
#   medium ~0.55-0.75  (more compliance, harder)
#   hard   ~0.40-0.60  (multi-country, conflicts)
# ---------------------------------------------------------------------------

_TASK_CEILING: dict[str, float] = {
    "easy":   0.949,  # strict upper bound (never exactly 1.0)
    "medium": 0.749,  # strict upper bound
    "hard":   0.599,  # strict upper bound
}

# ---------------------------------------------------------------------------
# Valid actions per task — used for parsimony penalty
# FIX 7: department/finalize actions stored WITHOUT target (matches to_key())
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
    "approve_hr",           # FIX 7: no target suffix
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
    "approve_hr",           # FIX 7
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
    "approve_hr",           # FIX 7
    "approve_legal",
    "approve_finance",
    "set_payroll",
    "set_tax_id",           # valid for Germany only
    "resolve_conflict",
    "finalize_case",
}

_VALID_ACTIONS_MAP: dict[str, set[str]] = {
    "easy":   _EASY_VALID_ACTIONS,
    "medium": _MEDIUM_VALID_ACTIONS,
    "hard":   _HARD_VALID_ACTIONS,
}


# ---------------------------------------------------------------------------
# Parsimony penalty
# ---------------------------------------------------------------------------

def _parsimony_penalty(prev_actions: list[str], valid_actions: set[str]) -> float:
    """
    -0.03 per action not in valid set. Capped at -0.15.
    """
    junk = sum(1 for a in prev_actions if a not in valid_actions)
    return min(0.15, junk * 0.03)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def grade(task_name: str, state: dict[str, Any]) -> float:
    """
    Main entry point. Called by environment.py after every episode end.

    Args:
        task_name: "easy" | "medium" | "hard"
        state:     Episode state dict.

    Returns:
        Float in [0.0, 1.0].
    """
    graders = {
        "easy":   grade_easy,
        "medium": grade_medium,
        "hard":   grade_hard,
    }
    if task_name not in graders:
        raise ValueError(
            f"Unknown task '{task_name}'. Valid: {list(graders.keys())}"
        )
    score = graders[task_name](state)
    # Final safety clamp — scores must always be in [0.0, 1.0]
    # Enforce STRICT bounds (0.0 and 1.0 both rejected by validator)
    score = max(0.001, min(0.999, score))
    assert 0.001 <= score <= 0.999, f"Score out of strict range: {score}"
    return score


def explain(task_name: str, state: dict[str, Any]) -> str:
    """Return a human-readable scoring breakdown string."""
    reporters = {
        "easy":   _explain_easy,
        "medium": _explain_medium,
        "hard":   _explain_hard,
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
#
# A perfect agent scores: 1.0 → clamped to 1.0
# Expected range: 0.70 – 1.00
# ---------------------------------------------------------------------------

def grade_easy(state: dict[str, Any]) -> float:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    status     = state.get("status", "in_progress")
    prev_acts  = state.get("previous_actions", [])

    required_docs  = _TASK_DOCS["easy"]
    required_comp  = _TASK_COMPLIANCE["easy"]

    # Documents (0.40)
    verified = sum(
        1 for d in required_docs
        if docs.get(d, {}).get("status") == "verified"
    )
    doc_score = (verified / len(required_docs)) * 0.40

    # HR approval (0.25)    FIX 3: only HR required for easy
    hr_score = 0.25 if depts.get("HR", False) else 0.0

    # Compliance (0.25)
    comp_done = sum(1 for c in required_comp if compliance.get(c, False))
    comp_score = (comp_done / len(required_comp)) * 0.25

    # Finalized (0.10)
    fin_score = 0.10 if status == "success" else 0.0

    raw = doc_score + hr_score + comp_score + fin_score

    # Parsimony penalty
    raw -= _parsimony_penalty(prev_acts, _EASY_VALID_ACTIONS)

    return round(max(0.0, min(_TASK_CEILING["easy"], raw)), 4)


# ---------------------------------------------------------------------------
# MEDIUM grader — India → Singapore
#
# Weights (sum = 1.0):
#   Documents verified    0.30   (3 docs, 0.10 each)
#   HR approved           0.15
#   Legal approved        0.20
#   Compliance            0.25   (payroll 0.08 + pdpa 0.09 + shadow 0.08)
#   Finalized             0.10
#
# A perfect agent scores: 1.0 → clamped to 1.0
# Expected range: 0.40 – 0.80
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

    # Legal (0.20)   FIX 4: medium needs Legal, not Finance
    legal_score = 0.20 if depts.get("Legal", False) else 0.0

    # Compliance (0.25): payroll + pdpa + shadow_payroll
    comp_done = sum(1 for c in required_comp if compliance.get(c, False))
    comp_score = (comp_done / len(required_comp)) * 0.25

    # Finalized (0.10)
    fin_score = 0.10 if status == "success" else 0.0

    raw = doc_score + hr_score + legal_score + comp_score + fin_score

    # Parsimony penalty
    raw -= _parsimony_penalty(prev_acts, _MEDIUM_VALID_ACTIONS)

    return round(max(0.0, min(_TASK_CEILING["medium"], raw)), 4)


# ---------------------------------------------------------------------------
# HARD grader — India → Germany + UAE
#
# Weights (sum = 1.0):
#   Documents verified    0.25   (4 docs, 0.0625 each)
#   HR approved           0.10
#   Legal approved        0.10
#   Finance approved      0.10
#   Compliance            0.15   (tax_id 0.075 + payroll 0.075)
#   UAE no-tax respected  0.20   (critical rule — NOT calling set_tax_id:UAE)
#   Conflict resolved     0.10
#   Finalized             0.10   (requires above all to be valid)
#                  TOTAL  1.10   → clamped to 1.0
#
# UAE no-tax rule: agent gets 0.20 for NOT calling set_tax_id:UAE
# If agent calls set_tax_id:UAE → loses 0.20 + gets extra -0.10 penalty
#
# Expected range: 0.20 – 0.60
# ---------------------------------------------------------------------------

def grade_hard(state: dict[str, Any]) -> float:
    """
    Hard grader — India → Germany + UAE.

    Calibrated so:
      - Empty state:   ~0.00
      - Perfect agent: ~0.55 (inside 0.20-0.60 range)
      - UAE tax violation: drops below 0.20

    Weights (sum = 0.90, ceiling = 0.60):
      Documents verified    0.25
      HR approved           0.08
      Legal approved        0.08
      Finance approved      0.08
      Compliance done       0.16  (tax_id 0.08 + payroll 0.08)
      Conflict resolved     0.15
      Finalized             0.10
                     TOTAL  0.90  → perfect raw = 0.90, capped at 0.60

    UAE trap penalty: -0.25 if set_tax_id:UAE called
    This gives UAE violators ~0.30 maximum (below 0.60 ceiling).
    """
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

    # Compliance (0.16 total — 0.08 per item)
    comp_done = sum(1 for c in required_comp if compliance.get(c, False))
    comp_score = (comp_done / len(required_comp)) * 0.16 if required_comp else 0.0

    # Conflict resolved (0.15)
    # Only award if conflicts actually exist AND are resolved
    if conflicts:
        all_resolved = all(c.get("resolved", False) for c in conflicts)
        conflict_score = 0.15 if all_resolved else 0.0
    else:
        conflict_score = 0.0  # no conflicts defined = no points

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

    # Hard ceiling: 0.60 — perfect agent scores 0.90 raw → capped at 0.60
    return round(max(0.0, min(0.60, raw)), 4)


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

    verified = [d for d in required_docs if docs.get(d, {}).get("status") == "verified"]
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

    verified      = [d for d in required_docs if docs.get(d, {}).get("status") == "verified"]
    comp_done     = [c for c in required_comp if compliance.get(c, False)]
    uae_tax_called = "set_tax_id:UAE" in prev_acts
    resolved      = all(c.get("resolved", False) for c in conflicts) if conflicts else True

    lines = [
        "=== HARD TASK SCORE BREAKDOWN ===",
        f"Documents:        {len(verified)}/{len(required_docs)} verified {verified}",
        f"HR:               {'✓' if depts.get('HR') else '✗'}",
        f"Legal:            {'✓' if depts.get('Legal') else '✗'}",
        f"Finance:          {'✓' if depts.get('Finance') else '✗'}",
        f"Compliance:       {len(comp_done)}/{len(required_comp)} done {comp_done}",
        f"UAE no-tax rule:  {'✗ VIOLATED (-0.30)' if uae_tax_called else '✓ respected (+0.20)'}",
        f"Conflicts:        {'✓ resolved' if resolved else '✗ unresolved'}",
        f"Status:           {status}",
        f"Score:            {grade_hard(state):.4f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch utility used by inference.py
# ---------------------------------------------------------------------------

def grade_all(states: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Grade all three tasks. Missing tasks default to 0.0."""
    results: dict[str, float] = {}
    for task_name in ["easy", "medium", "hard"]:
        if task_name in states:
            try:
                results[task_name] = grade(task_name, states[task_name])
            except Exception as exc:
                results[task_name] = 0.0
                print(f"[grader] ERROR grading '{task_name}': {exc}")
        else:
            results[task_name] = 0.0
    return results