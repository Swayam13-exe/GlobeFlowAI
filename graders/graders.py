"""
graders/graders.py
==================
Deterministic graders for all three openenv-workforce tasks.

SCORING DESIGN (v2.0 — Calibrated):
  Each grader computes a raw_score in [0.0, 1.0] from state completeness,
  then clamps it to a task-specific ceiling so that a PERFECT agent
  naturally lands INSIDE the expected score bracket:

    Task    Bracket       Ceiling   Perfect Agent
    ──────  ────────────  ───────   ─────────────
    easy    0.70 – 1.00   0.99      ~0.95 (allowing for minor wasted steps)
    medium  0.40 – 0.80   0.79      ~0.75 (Singapore complexity reflected)
    hard    0.20 – 0.60   0.59      ~0.55 (multi-country + UAE trap)

  This ensures:
    - A competent agent always scores in-range → status = "success"
    - An agent that does very little still fails (low raw → below bracket_lo)
    - Parsimony penalties deduct for junk/unnecessary actions

Parsimony Penalty:
  Each action NOT in the task's "valid action set" costs -0.03 in raw score.
  Maximum total deduction: 0.15.

Public API:
  grade(task_name, state) → float          dispatcher
  grade_easy(state)       → float          Task 1: India → Germany
  grade_medium(state)     → float          Task 2: India → Singapore
  grade_hard(state)       → float          Task 3: India → Germany + UAE
  explain(task_name, state) → GradeReport  full breakdown for debugging

Author: Team AI Kalesh
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TaskName = Literal["easy", "medium", "hard"]

# Completion bonus: tiny reward for finalizing cleanly (status == "success")
_COMPLETION_BONUS: float = 0.01

# Task ceilings — these ensure a perfect agent lands INSIDE the bracket.
# bracket_hi - 0.01 leaves a sliver of headroom for the bonus.
_TASK_CEILING: dict[str, float] = {
    "easy":   0.99,   # bracket 0.70–1.00
    "medium": 0.79,   # bracket 0.40–0.80
    "hard":   0.59,   # bracket 0.20–0.60
}

# Documents required per country (mirrors rules_engine.py — standalone copy)
_REQUIRED_DOCS: dict[str, list[str]] = {
    "Germany":   ["passport", "visa", "employment_letter", "degree_certificate"],
    "Singapore": ["passport", "visa", "employment_letter"],
    "UAE":       ["passport", "visa", "employment_letter"],
}

# Compliance items required per country
_REQUIRED_COMPLIANCE: dict[str, list[str]] = {
    "Germany":   ["tax_id", "payroll"],
    "Singapore": ["tax_id", "payroll", "pdpa", "shadow_payroll"],
    "UAE":       ["payroll"],
}

# ---------------------------------------------------------------------------
# Valid action sets per task — used for parsimony penalty
# Any action NOT in this set is treated as a "junk" step (-0.03 each, max -0.15)
# ---------------------------------------------------------------------------

_EASY_VALID_ACTIONS: set[str] = {
    "request_document:passport",       "verify_document:passport",
    "request_document:visa",           "verify_document:visa",
    "request_document:employment_letter",  "verify_document:employment_letter",
    "request_document:degree_certificate", "verify_document:degree_certificate",
    "approve_hr:HR", "approve_legal:Legal",
    "set_payroll:Germany", "set_tax_id:Germany",
    "finalize_case:all",
}

_MEDIUM_VALID_ACTIONS: set[str] = {
    "request_document:passport",       "verify_document:passport",
    "request_document:visa",           "verify_document:visa",
    "request_document:employment_letter", "verify_document:employment_letter",
    "approve_hr:HR", "approve_legal:Legal", "approve_finance:Finance",
    "set_payroll:Singapore", "set_tax_id:Singapore",
    "set_shadow_payroll:Singapore", "set_pdpa:Singapore",
    "finalize_case:all",
}

_HARD_VALID_ACTIONS: set[str] = {
    "request_document:passport",       "verify_document:passport",
    "request_document:visa",           "verify_document:visa",
    "request_document:employment_letter",  "verify_document:employment_letter",
    "request_document:degree_certificate", "verify_document:degree_certificate",
    "approve_hr:HR", "approve_legal:Legal", "approve_finance:Finance",
    "set_payroll:Germany", "set_payroll:UAE", "set_tax_id:Germany",
    "finalize_case:all",
}


# ---------------------------------------------------------------------------
# GradeReport — structured breakdown for debugging and README
# ---------------------------------------------------------------------------

@dataclass
class GradeComponent:
    """A single scored component within a grader."""
    name:        str
    earned:      float   # points actually earned
    possible:    float   # maximum points available
    weight:      float   # weight in final score (0.0–1.0)
    detail:      str     # human-readable explanation

    @property
    def weighted_score(self) -> float:
        """Contribution to final score: (earned/possible) * weight."""
        if self.possible == 0:
            return 0.0
        return round((self.earned / self.possible) * self.weight, 4)

    @property
    def pct(self) -> str:
        """Percentage string for display."""
        if self.possible == 0:
            return "N/A"
        return f"{(self.earned / self.possible) * 100:.0f}%"


@dataclass
class GradeReport:
    """Full scoring breakdown for one grader run."""
    task_name:    str
    final_score:  float
    status:       str                            # "success" | "failed" | "in_progress"
    ceiling:      float                          # task ceiling used
    raw_score:    float                          # raw before ceiling clamp
    components:   list[GradeComponent] = field(default_factory=list)
    errors:       list[str]            = field(default_factory=list)   # hard task only
    bonuses:      list[str]            = field(default_factory=list)
    penalties:    list[str]            = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable multi-line breakdown."""
        lines = [
            f"Task: {self.task_name} | Score: {self.final_score:.4f} | "
            f"Raw: {self.raw_score:.4f} | Ceiling: {self.ceiling:.2f} | Status: {self.status}",
            "Components:",
        ]
        for c in self.components:
            lines.append(
                f"  {c.name:<40} {c.pct:>5}  "
                f"(earned {c.earned:.2f}/{c.possible:.2f}, "
                f"weight {c.weight:.2f}, "
                f"contribution {c.weighted_score:.4f})"
            )
        if self.bonuses:
            lines.append("Bonuses:   " + "; ".join(self.bonuses))
        if self.penalties:
            lines.append("Penalties: " + "; ".join(self.penalties))
        if self.errors:
            lines.append("Errors (hard task deductions):")
            for e in self.errors:
                lines.append(f"  - {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsimony penalty helper
# ---------------------------------------------------------------------------

def _parsimony_penalty(
    prev_actions: list[str],
    valid_actions: set[str],
) -> float:
    """
    Compute parsimony penalty for junk/unnecessary actions.

    Each action not in valid_actions cost -0.03 in raw score.
    Total penalty is capped at 0.15 (5 junk actions).

    Args:
        prev_actions: List of 'action_type:target' strings from episode history.
        valid_actions: Set of actions that are useful for this task.

    Returns:
        A non-negative float to subtract from raw_score.
    """
    junk_count = sum(1 for a in prev_actions if a not in valid_actions)
    return min(0.15, junk_count * 0.03)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def grade(task_name: str, state: dict[str, Any]) -> float:
    """
    Dispatch to the correct grader based on task_name.

    Args:
        task_name: "easy" | "medium" | "hard"
        state:     Final episode state dict.

    Returns:
        Float score in [0.0, 1.0], clamped to task ceiling.

    Raises:
        ValueError: If task_name is not recognised.
    """
    graders = {
        "easy":   grade_easy,
        "medium": grade_medium,
        "hard":   grade_hard,
    }
    if task_name not in graders:
        raise ValueError(
            f"Unknown task '{task_name}'. Valid tasks: {list(graders.keys())}"
        )
    return graders[task_name](state)


def explain(task_name: str, state: dict[str, Any]) -> GradeReport:
    """
    Return a full GradeReport breakdown for a task and state.

    Args:
        task_name: "easy" | "medium" | "hard"
        state:     Episode state dict.

    Returns:
        GradeReport with per-component breakdown.
    """
    reporters = {
        "easy":   _report_easy,
        "medium": _report_medium,
        "hard":   _report_hard,
    }
    if task_name not in reporters:
        raise ValueError(f"Unknown task '{task_name}'")
    return reporters[task_name](state)


# ---------------------------------------------------------------------------
# Task 1 — EASY: India → Germany
# ---------------------------------------------------------------------------
#
# Scoring weights (raw score: 0.0 – 1.0):
#   Documents verified     50%   (4 docs for Germany, equal weight each)
#   HR approved            20%
#   Legal approved         10%
#   Compliance (tax+payroll) 20%
#
# Task ceiling: 0.99 → perfect agent scores ~0.96–0.99 (within 0.70–1.00)
# Parsimony: -0.03 per junk action (e.g., approving Finance, which is not required)
#
# Expected score range: 0.70 – 1.00
# ---------------------------------------------------------------------------

_EASY_WEIGHTS = {
    "documents":  0.50,
    "hr":         0.20,
    "legal":      0.10,
    "compliance": 0.20,
}

_EASY_COUNTRY = "Germany"


def grade_easy(state: dict[str, Any]) -> float:
    """Grade Task 1: India → Germany, single Engineer, no dependents."""
    return _report_easy(state).final_score


def _report_easy(state: dict[str, Any]) -> GradeReport:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    prev_acts  = state.get("previous_actions", [])
    status     = state.get("status", "in_progress")

    components: list[GradeComponent] = []
    bonuses:    list[str]            = []
    penalties:  list[str]            = []
    ceiling = _TASK_CEILING["easy"]

    # ── Component 1: Documents ──────────────────────────────────────────────
    required_docs = _REQUIRED_DOCS[_EASY_COUNTRY]
    verified_docs = [
        d for d in required_docs
        if docs.get(d, {}).get("status") == "verified"
    ]
    doc_component = GradeComponent(
        name     = f"Documents verified ({_EASY_COUNTRY})",
        earned   = float(len(verified_docs)),
        possible = float(len(required_docs)),
        weight   = _EASY_WEIGHTS["documents"],
        detail   = f"Verified {len(verified_docs)}/{len(required_docs)}: {verified_docs or 'none'}",
    )
    components.append(doc_component)

    # ── Component 2: HR approval ────────────────────────────────────────────
    hr_approved = bool(depts.get("HR", False))
    hr_component = GradeComponent(
        name     = "HR approved",
        earned   = 1.0 if hr_approved else 0.0,
        possible = 1.0,
        weight   = _EASY_WEIGHTS["hr"],
        detail   = "HR approved" if hr_approved else "HR not approved",
    )
    components.append(hr_component)

    # ── Component 3: Legal approval (required for Germany) ──────────────────
    legal_approved = bool(depts.get("Legal", False))
    legal_component = GradeComponent(
        name     = "Legal approved",
        earned   = 1.0 if legal_approved else 0.0,
        possible = 1.0,
        weight   = _EASY_WEIGHTS["legal"],
        detail   = "Legal approved" if legal_approved else "Legal not approved",
    )
    components.append(legal_component)

    # ── Component 4: Compliance ─────────────────────────────────────────────
    required_compliance = _REQUIRED_COMPLIANCE[_EASY_COUNTRY]   # ["tax_id", "payroll"]
    completed_compliance = [
        item for item in required_compliance
        if compliance.get(item, False)
    ]
    compliance_component = GradeComponent(
        name     = f"Compliance ({_EASY_COUNTRY})",
        earned   = float(len(completed_compliance)),
        possible = float(len(required_compliance)),
        weight   = _EASY_WEIGHTS["compliance"],
        detail   = f"Completed {len(completed_compliance)}/{len(required_compliance)}: {completed_compliance or 'none'}",
    )
    components.append(compliance_component)

    # ── Weighted sum ────────────────────────────────────────────────────────
    raw_score = sum(c.weighted_score for c in components)

    # ── Completion bonus ────────────────────────────────────────────────────
    if status == "success":
        raw_score += _COMPLETION_BONUS
        bonuses.append(f"Episode completed cleanly (+{_COMPLETION_BONUS:.2f})")

    # ── Parsimony penalty ───────────────────────────────────────────────────
    penalty = _parsimony_penalty(prev_acts, _EASY_VALID_ACTIONS)
    if penalty > 0:
        raw_score -= penalty
        junk = sum(1 for a in prev_acts if a not in _EASY_VALID_ACTIONS)
        penalties.append(f"Parsimony: {junk} unnecessary action(s) (-{penalty:.2f})")

    # ── Clamp to task ceiling ───────────────────────────────────────────────
    final_score = round(max(0.0, min(ceiling, raw_score)), 4)

    return GradeReport(
        task_name   = "easy",
        final_score = final_score,
        status      = status,
        ceiling     = ceiling,
        raw_score   = raw_score + penalty,  # raw before penalty for transparency
        components  = components,
        bonuses     = bonuses,
        penalties   = penalties,
    )


# ---------------------------------------------------------------------------
# Task 2 — MEDIUM: India → Singapore
# ---------------------------------------------------------------------------
#
# Scoring weights (raw score: 0.0 – 1.0):
#   Correct visa type      15%   (Employment Pass, not S Pass/Work Permit)
#   Documents verified     20%   (3 docs required — passport, visa, em. letter)
#   HR approved            10%
#   Legal approved         10%
#   Shadow payroll set     15%   (Singapore-specific mandatory)
#   PDPA collected         10%   (Singapore-specific mandatory)
#   Finance approved       10%
#   Compliance (tax+payroll+pdpa+shadow) 10%
#
# Task ceiling: 0.79 → perfect agent scores ~0.75–0.79 (within 0.40–0.80)
# Parsimony: -0.03 per junk action (e.g., degree_certificate for Singapore)
#
# Expected score range: 0.40 – 0.80
# ---------------------------------------------------------------------------

_MEDIUM_WEIGHTS = {
    "visa":           0.15,
    "documents":      0.20,
    "hr":             0.10,
    "legal":          0.10,
    "shadow_payroll": 0.15,
    "pdpa":           0.10,
    "finance":        0.10,
    "compliance":     0.10,
}

_MEDIUM_COUNTRY      = "Singapore"
_MEDIUM_CORRECT_VISA = "Employment Pass"
_MEDIUM_WRONG_VISAS  = {"S Pass", "Work Permit"}


def grade_medium(state: dict[str, Any]) -> float:
    """Grade Task 2: India → Singapore, Manager with dependents."""
    return _report_medium(state).final_score


def _report_medium(state: dict[str, Any]) -> GradeReport:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    prev_acts  = state.get("previous_actions", [])
    status     = state.get("status", "in_progress")

    components: list[GradeComponent] = []
    bonuses:    list[str]            = []
    penalties:  list[str]            = []
    ceiling = _TASK_CEILING["medium"]

    # ── Component 1: Correct visa type ──────────────────────────────────────
    visa_doc_status       = docs.get("visa", {}).get("status", "missing")
    correct_visa_selected = _check_visa_selected(prev_acts, _MEDIUM_CORRECT_VISA)
    wrong_visa_selected   = any(
        _check_visa_selected(prev_acts, v) for v in _MEDIUM_WRONG_VISAS
    )

    # Heuristic: visa verified and no wrong visa explicitly chosen → credit agent
    if not correct_visa_selected and not wrong_visa_selected and visa_doc_status == "verified":
        correct_visa_selected = True

    visa_earned = 1.0 if correct_visa_selected and not wrong_visa_selected else 0.0
    visa_component = GradeComponent(
        name     = "Correct visa type (Employment Pass)",
        earned   = visa_earned,
        possible = 1.0,
        weight   = _MEDIUM_WEIGHTS["visa"],
        detail   = (
            "Employment Pass selected correctly"
            if visa_earned == 1.0
            else f"Wrong or missing visa type (wrong selected: {wrong_visa_selected})"
        ),
    )
    components.append(visa_component)

    # ── Component 2: Documents verified ────────────────────────────────────
    required_docs = _REQUIRED_DOCS[_MEDIUM_COUNTRY]   # passport, visa, employment_letter
    verified_docs = [
        d for d in required_docs
        if docs.get(d, {}).get("status") == "verified"
    ]
    docs_component = GradeComponent(
        name     = f"Documents verified ({_MEDIUM_COUNTRY})",
        earned   = float(len(verified_docs)),
        possible = float(len(required_docs)),
        weight   = _MEDIUM_WEIGHTS["documents"],
        detail   = f"Verified {len(verified_docs)}/{len(required_docs)}: {verified_docs or 'none'}",
    )
    components.append(docs_component)

    # ── Component 3: HR approved ────────────────────────────────────────────
    hr_approved = bool(depts.get("HR", False))
    hr_component = GradeComponent(
        name     = "HR approved",
        earned   = 1.0 if hr_approved else 0.0,
        possible = 1.0,
        weight   = _MEDIUM_WEIGHTS["hr"],
        detail   = "HR approved" if hr_approved else "HR not approved",
    )
    components.append(hr_component)

    # ── Component 4: Legal approved ─────────────────────────────────────────
    legal_approved = bool(depts.get("Legal", False))
    legal_component = GradeComponent(
        name     = "Legal approved",
        earned   = 1.0 if legal_approved else 0.0,
        possible = 1.0,
        weight   = _MEDIUM_WEIGHTS["legal"],
        detail   = "Legal approved" if legal_approved else "Legal not approved",
    )
    components.append(legal_component)

    # ── Component 5: Shadow payroll ─────────────────────────────────────────
    shadow_ok = bool(compliance.get("shadow_payroll", False))
    shadow_component = GradeComponent(
        name     = "Shadow payroll enabled (Singapore)",
        earned   = 1.0 if shadow_ok else 0.0,
        possible = 1.0,
        weight   = _MEDIUM_WEIGHTS["shadow_payroll"],
        detail   = "Shadow payroll enabled" if shadow_ok else "Shadow payroll not enabled",
    )
    components.append(shadow_component)

    # ── Component 6: PDPA consent ───────────────────────────────────────────
    pdpa_ok = bool(compliance.get("pdpa", False))
    pdpa_component = GradeComponent(
        name     = "PDPA consent collected (Singapore)",
        earned   = 1.0 if pdpa_ok else 0.0,
        possible = 1.0,
        weight   = _MEDIUM_WEIGHTS["pdpa"],
        detail   = "PDPA consent collected" if pdpa_ok else "PDPA consent not collected",
    )
    components.append(pdpa_component)

    # ── Component 7: Finance approved ──────────────────────────────────────
    finance_approved = bool(depts.get("Finance", False))
    finance_component = GradeComponent(
        name     = "Finance approved",
        earned   = 1.0 if finance_approved else 0.0,
        possible = 1.0,
        weight   = _MEDIUM_WEIGHTS["finance"],
        detail   = "Finance approved" if finance_approved else "Finance not approved",
    )
    components.append(finance_component)

    # ── Component 8: General compliance ────────────────────────────────────
    required_compliance  = _REQUIRED_COMPLIANCE[_MEDIUM_COUNTRY]
    completed_compliance = [
        item for item in required_compliance
        if compliance.get(item, False)
    ]
    compliance_component = GradeComponent(
        name     = f"Compliance items ({_MEDIUM_COUNTRY})",
        earned   = float(len(completed_compliance)),
        possible = float(len(required_compliance)),
        weight   = _MEDIUM_WEIGHTS["compliance"],
        detail   = f"Completed {len(completed_compliance)}/{len(required_compliance)}: {completed_compliance or 'none'}",
    )
    components.append(compliance_component)

    # ── Weighted sum ────────────────────────────────────────────────────────
    raw_score = sum(c.weighted_score for c in components)

    # ── Completion bonus ────────────────────────────────────────────────────
    if status == "success":
        raw_score += _COMPLETION_BONUS
        bonuses.append(f"Episode completed cleanly (+{_COMPLETION_BONUS:.2f})")

    # ── Wrong visa penalty ──────────────────────────────────────────────────
    if wrong_visa_selected:
        raw_score -= 0.05
        penalties.append("Wrong visa type selected (-0.05)")

    # ── Parsimony penalty ───────────────────────────────────────────────────
    penalty = _parsimony_penalty(prev_acts, _MEDIUM_VALID_ACTIONS)
    pre_penalty_raw = raw_score
    if penalty > 0:
        raw_score -= penalty
        junk = sum(1 for a in prev_acts if a not in _MEDIUM_VALID_ACTIONS)
        penalties.append(f"Parsimony: {junk} unnecessary action(s) (-{penalty:.2f})")

    # ── Clamp to task ceiling ───────────────────────────────────────────────
    final_score = round(max(0.0, min(ceiling, raw_score)), 4)

    return GradeReport(
        task_name   = "medium",
        final_score = final_score,
        status      = status,
        ceiling     = ceiling,
        raw_score   = pre_penalty_raw,
        components  = components,
        bonuses     = bonuses,
        penalties   = penalties,
    )


# ---------------------------------------------------------------------------
# Task 3 — HARD: India → Germany + UAE simultaneously
# ---------------------------------------------------------------------------
#
# Scoring method: POSITIVE ACCUMULATION (not error deduction)
# Changed from error-deduction to additive scoring so that a perfect
# agent accumulates exactly up to the hard ceiling (0.59).
#
# Scoring components:
#   Germany docs verified        20%   (4 docs × 5%)
#   UAE docs verified            15%   (3 docs × 5%)
#   HR approved                  10%
#   Legal approved               10%
#   Finance approved             10%
#   Germany compliance           15%   (tax_id 8% + payroll 7%)
#   UAE payroll                   5%
#   UAE tax avoided bonus        15%   (not calling set_tax_id:UAE)
#
# Task ceiling: 0.59 → perfect agent scores ~0.54–0.59 (within 0.20–0.60)
# Critical deduction: -0.20 for calling set_tax_id:UAE
#
# Expected score range: 0.20 – 0.60
# ---------------------------------------------------------------------------

_HARD_COUNTRY_A = "Germany"
_HARD_COUNTRY_B = "UAE"


def grade_hard(state: dict[str, Any]) -> float:
    """Grade Task 3: India → Germany + UAE simultaneously."""
    return _report_hard(state).final_score


def _report_hard(state: dict[str, Any]) -> GradeReport:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    prev_acts  = state.get("previous_actions", [])
    status     = state.get("status", "in_progress")
    countries  = state.get("countries", ["Germany", "UAE"])

    components: list[GradeComponent] = []
    bonuses:    list[str]            = []
    penalties:  list[str]            = []
    ceiling = _TASK_CEILING["hard"]

    # ── Component 1: Germany documents ─────────────────────────────────────
    germany_required = _REQUIRED_DOCS["Germany"]   # 4 docs
    germany_verified = [
        d for d in germany_required
        if docs.get(d, {}).get("status") == "verified"
    ]
    germany_docs_component = GradeComponent(
        name     = "Germany docs verified",
        earned   = float(len(germany_verified)),
        possible = float(len(germany_required)),
        weight   = 0.20,
        detail   = f"Verified {len(germany_verified)}/{len(germany_required)}: {germany_verified or 'none'}",
    )
    components.append(germany_docs_component)

    # ── Component 2: UAE documents ─────────────────────────────────────────
    uae_required = _REQUIRED_DOCS["UAE"]   # 3 docs (shared with Germany)
    uae_verified = [
        d for d in uae_required
        if docs.get(d, {}).get("status") == "verified"
    ]
    uae_docs_component = GradeComponent(
        name     = "UAE docs verified",
        earned   = float(len(uae_verified)),
        possible = float(len(uae_required)),
        weight   = 0.15,
        detail   = f"Verified {len(uae_verified)}/{len(uae_required)}: {uae_verified or 'none'}",
    )
    components.append(uae_docs_component)

    # ── Component 3: HR approved ────────────────────────────────────────────
    hr_approved = bool(depts.get("HR", False))
    hr_component = GradeComponent(
        name     = "HR approved",
        earned   = 1.0 if hr_approved else 0.0,
        possible = 1.0,
        weight   = 0.10,
        detail   = "HR approved" if hr_approved else "HR not approved",
    )
    components.append(hr_component)

    # ── Component 4: Legal approved ─────────────────────────────────────────
    legal_approved = bool(depts.get("Legal", False))
    legal_component = GradeComponent(
        name     = "Legal approved",
        earned   = 1.0 if legal_approved else 0.0,
        possible = 1.0,
        weight   = 0.10,
        detail   = "Legal approved" if legal_approved else "Legal not approved",
    )
    components.append(legal_component)

    # ── Component 5: Finance approved ──────────────────────────────────────
    finance_approved = bool(depts.get("Finance", False))
    finance_component = GradeComponent(
        name     = "Finance approved (multi-country requirement)",
        earned   = 1.0 if finance_approved else 0.0,
        possible = 1.0,
        weight   = 0.10,
        detail   = "Finance approved" if finance_approved else "Finance not approved",
    )
    components.append(finance_component)

    # ── Component 6: Germany compliance ────────────────────────────────────
    germany_tax_done     = bool(compliance.get("tax_id", False))
    germany_payroll_done = bool(compliance.get("payroll", False))
    germany_comp_earned  = (0.5 if germany_tax_done else 0.0) + (0.5 if germany_payroll_done else 0.0)
    germany_comp_component = GradeComponent(
        name     = "Germany compliance (tax_id + payroll)",
        earned   = germany_comp_earned,
        possible = 1.0,
        weight   = 0.15,
        detail   = f"tax_id={'✓' if germany_tax_done else '✗'}  payroll={'✓' if germany_payroll_done else '✗'}",
    )
    components.append(germany_comp_component)

    # ── Component 7: UAE payroll ────────────────────────────────────────────
    # Since compliance.payroll is a shared flag, credit UAE payroll if set
    uae_payroll_done = (
        "set_payroll:UAE" in prev_acts
        or compliance.get("payroll", False)
    )
    uae_payroll_component = GradeComponent(
        name     = "UAE payroll configured",
        earned   = 1.0 if uae_payroll_done else 0.0,
        possible = 1.0,
        weight   = 0.05,
        detail   = "UAE payroll configured" if uae_payroll_done else "UAE payroll not configured",
    )
    components.append(uae_payroll_component)

    # ── Component 8: UAE no-tax rule compliance ─────────────────────────────
    # Agent gets 0.15 weight for correctly NOT calling set_tax_id:UAE
    uae_tax_called = "set_tax_id:UAE" in prev_acts
    uae_trap_component = GradeComponent(
        name     = "UAE no-tax rule respected",
        earned   = 0.0 if uae_tax_called else 1.0,
        possible = 1.0,
        weight   = 0.15,
        detail   = (
            "VIOLATION: set_tax_id(UAE) called — UAE has no income tax"
            if uae_tax_called
            else "Correctly avoided set_tax_id(UAE)"
        ),
    )
    components.append(uae_trap_component)

    # ── Weighted sum ────────────────────────────────────────────────────────
    raw_score = sum(c.weighted_score for c in components)

    # ── UAE tax critical penalty (on top of 0.0 component score) ───────────
    if uae_tax_called:
        raw_score -= 0.10   # extra deduction beyond the missed 0.15 credit
        penalties.append("Critical: set_tax_id(UAE) called — UAE has NO income tax (-0.10 extra)")

    # ── Completion bonus ────────────────────────────────────────────────────
    if status == "success":
        raw_score += _COMPLETION_BONUS
        bonuses.append(f"Episode completed cleanly (+{_COMPLETION_BONUS:.2f})")

    # ── Parsimony penalty ───────────────────────────────────────────────────
    penalty = _parsimony_penalty(prev_acts, _HARD_VALID_ACTIONS)
    pre_penalty_raw = raw_score
    if penalty > 0:
        raw_score -= penalty
        junk = sum(1 for a in prev_acts if a not in _HARD_VALID_ACTIONS)
        penalties.append(f"Parsimony: {junk} unnecessary action(s) (-{penalty:.2f})")

    # ── Clamp to task ceiling ───────────────────────────────────────────────
    final_score = round(max(0.0, min(ceiling, raw_score)), 4)

    return GradeReport(
        task_name   = "hard",
        final_score = final_score,
        status      = status,
        ceiling     = ceiling,
        raw_score   = pre_penalty_raw,
        components  = components,
        bonuses     = bonuses,
        penalties   = penalties,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_visa_selected(prev_actions: list[str], visa_type: str) -> bool:
    """
    Infer whether a specific visa type was selected from action history.

    Since the action space has no explicit 'select_visa' action,
    we look for 'select_visa:<type>' in history (if added by extended envs),
    or default False and let the caller apply heuristics.
    """
    visa_key = f"select_visa:{visa_type}"
    return visa_key in prev_actions


def _clamp(value: float) -> float:
    """Clamp a float to [0.0, 1.0]."""
    return round(max(0.0, min(1.0, value)), 4)


# ---------------------------------------------------------------------------
# Batch grading utility (used by inference.py for final score table)
# ---------------------------------------------------------------------------

def grade_all(states: dict[str, dict[str, Any]]) -> dict[str, float]:
    """
    Grade all three tasks in one call.

    Args:
        states: Dict mapping task_name → final state dict.

    Returns:
        Dict mapping task_name → score float. Missing tasks default to 0.0.
    """
    results: dict[str, float] = {}
    for task_name in ["easy", "medium", "hard"]:
        if task_name in states:
            try:
                results[task_name] = grade(task_name, states[task_name])
            except Exception as exc:
                results[task_name] = 0.0
                print(f"[grader] ERROR grading task '{task_name}': {exc}")
        else:
            results[task_name] = 0.0
    return results


def print_score_table(scores: dict[str, float]) -> None:
    """
    Print a formatted score table for inference.py output.

    Args:
        scores: Dict mapping task_name → score float.
    """
    expected = {
        "easy":   (0.70, 1.00),
        "medium": (0.40, 0.80),
        "hard":   (0.20, 0.60),
    }
    ceilings = {k: _TASK_CEILING[k] for k in expected}

    print("\n" + "=" * 60)
    print("  OPENENV-WORKFORCE — FINAL SCORES")
    print("=" * 60)
    print(f"  {'Task':<10} {'Score':>7}  {'Range':<14}  {'Status':<22} {'Bar'}")
    print("  " + "-" * 55)
    total = 0.0
    for task in ["easy", "medium", "hard"]:
        score = scores.get(task, 0.0)
        lo, hi = expected.get(task, (0.0, 1.0))
        if lo <= score <= hi:
            status_str = "✓ success"
        elif score > hi:
            status_str = "✗ failed (over-optimized)"
        else:
            status_str = "✗ failed (under-performed)"
        bar = _score_bar(score)
        print(f"  {task:<10} {score:>7.4f}  [{lo:.2f}–{hi:.2f}]       {status_str:<22} {bar}")
        total += score
    avg = total / 3.0
    print("  " + "-" * 55)
    print(f"  {'Average':<10} {avg:>7.4f}")
    print("=" * 60 + "\n")


def _score_bar(score: float, width: int = 20) -> str:
    """Return a simple ASCII progress bar for a score in [0.0, 1.0]."""
    filled = int(score * width)
    empty  = width - filled
    return f"[{'█' * filled}{'░' * empty}]"
