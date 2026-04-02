"""
env/reward.py
=============
Reward computation for openenv-workforce.

Design principles:
  - All reward values clamped to [-1.0, 1.0] per step
  - Cumulative episode score clamped to [0.0, 1.0]
  - No randomness — same (action, result, state) always returns same reward
  - Progress delta bonus incentivises efficient completion
  - Milestone bonuses for completing entire document/department/compliance groups

REWARDS lookup table is imported by environment.py for quick penalty access.
compute_reward() is the main entry point used after every step().

Author: Team AI Kalesh
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Reward constants — imported by environment.py for fast lookups
# ---------------------------------------------------------------------------

REWARDS: dict[str, float] = {
    # ── Positive rewards ────────────────────────────────────────────────────
    "success":               0.4,   # action resolved cleanly (success result)
    "milestone_docs":        0.1,   # all required documents verified
    "milestone_depts":       0.1,   # all required departments approved
    "milestone_compliance":  0.1,   # all compliance items complete
    "episode_complete":      0.3,   # finalize_case called cleanly (done=True)
    "progress_delta":        0.05,  # per 0.25 progress increase (scaled below)

    # ── Negative rewards — penalties ────────────────────────────────────────
    "wrong_action":          -0.1,  # valid type, wrong context
    "prereq_violated":       -0.2,  # prerequisite not met
    "rule_violation":        -0.3,  # breaks a country rule (e.g. UAE tax)
    "invalid_action":        -0.4,  # unknown action_type or malformed target
    "repeated_action":       -0.1,  # same action called again this episode
}


# ---------------------------------------------------------------------------
# Result → base reward mapping
# ---------------------------------------------------------------------------

_RESULT_BASE: dict[str, float] = {
    "success":          REWARDS["success"],
    "wrong_action":     REWARDS["wrong_action"],
    "prereq_violated":  REWARDS["prereq_violated"],
    "rule_violation":   REWARDS["rule_violation"],
    "invalid_action":   REWARDS["invalid_action"],
    "repeated_action":  REWARDS["repeated_action"],
}


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------


def compute_reward(
    action: Any,          # env.models.Action  (typed as Any to avoid circular import)
    result: str,          # result code from action handler
    state: dict[str, Any],
    prev_progress: float,
) -> float:
    """
    Compute the per-step reward for a completed action.

    Steps:
      1. Base reward from result code (see REWARDS / _RESULT_BASE)
      2. Progress delta bonus — reward proportional to checklist progress made
      3. Milestone bonuses — extra reward for completing full groups
      4. Episode completion bonus — if result=="success" and state["status"]=="success"
      5. Clamp to [-1.0, 1.0]

    Args:
        action:        The Action model that was just executed.
        result:        Result string from the action handler
                       ("success" | "wrong_action" | "rule_violation" |
                        "prereq_violated" | "invalid_action" | "repeated_action")
        state:         Current (already-updated) episode state dict.
        prev_progress: state["progress"] value BEFORE this action was applied.

    Returns:
        Float clamped to [-1.0, 1.0].
    """
    # ── 1. Base reward ───────────────────────────────────────────────────────
    reward = _RESULT_BASE.get(result, 0.0)

    # Only apply bonuses/progress delta on successful actions
    if result == "success":

        # ── 2. Progress delta bonus ──────────────────────────────────────────
        new_progress = state.get("progress", prev_progress)
        delta = new_progress - prev_progress
        if delta > 0:
            # Scale: each 0.25 of progress gained adds ~0.05
            reward += round(delta * REWARDS["progress_delta"] * 4, 4)

        # ── 3. Milestone bonuses ─────────────────────────────────────────────
        reward += _check_milestones(action, state)

        # ── 4. Episode completion bonus ──────────────────────────────────────
        if (
            action.action_type == "finalize_case"
            and state.get("status") == "success"
        ):
            reward += REWARDS["episode_complete"]

    # ── 5. Clamp ─────────────────────────────────────────────────────────────
    return round(max(-1.0, min(1.0, reward)), 4)


# ---------------------------------------------------------------------------
# Milestone checker — called only on successful actions
# ---------------------------------------------------------------------------


def _check_milestones(action: Any, state: dict[str, Any]) -> float:
    """
    Return bonus reward if a major milestone was just reached.

    Milestones:
      - All required documents verified (for all countries in case)
      - All required departments approved
      - All required compliance items complete

    Each milestone pays once per episode (checked by whether the triggering
    action type is consistent with having just completed the group).

    Args:
        action: The Action model (used to gate which milestone to check).
        state:  Current state dict (already updated).

    Returns:
        Bonus reward float (0.0 if no milestone reached).
    """
    bonus = 0.0
    atype = action.action_type
    countries = state.get("countries", [])

    # ── Document milestone ───────────────────────────────────────────────────
    if atype == "verify_document":
        if _all_docs_verified(state, countries):
            bonus += REWARDS["milestone_docs"]

    # ── Department milestone ─────────────────────────────────────────────────
    if atype in {"approve_hr", "approve_legal", "approve_finance"}:
        depts = state.get("departments", {})
        if depts.get("HR") and depts.get("Legal") and depts.get("Finance"):
            bonus += REWARDS["milestone_depts"]
        elif depts.get("HR") and depts.get("Legal") and not _needs_finance(countries):
            # Two-dept case (Germany easy task): HR+Legal = done
            bonus += REWARDS["milestone_depts"]

    # ── Compliance milestone ─────────────────────────────────────────────────
    if atype in {"set_payroll", "set_tax_id", "set_shadow_payroll", "set_pdpa"}:
        if _all_compliance_done(state, countries):
            bonus += REWARDS["milestone_compliance"]

    return round(bonus, 4)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _all_docs_verified(state: dict[str, Any], countries: list[str]) -> bool:
    """True when all required documents for all case countries are verified."""
    # Import inline to avoid circular dependency
    from env.rules_engine import get_required_documents

    docs = state.get("documents", {})
    for country in countries:
        for doc_name in get_required_documents(country):
            if docs.get(doc_name, {}).get("status") != "verified":
                return False
    return True


def _all_compliance_done(state: dict[str, Any], countries: list[str]) -> bool:
    """True when all required compliance items for all case countries are complete."""
    from env.rules_engine import get_required_compliance

    compliance = state.get("compliance", {})
    for country in countries:
        for item in get_required_compliance(country):
            if not compliance.get(item, False):
                return False
    return True


def _needs_finance(countries: list[str]) -> bool:
    """True when Finance approval is required for this country combination."""
    from env.rules_engine import _load_country_rules
    rules = _load_country_rules()

    if len(countries) > 1:
        return True

    for country in countries:
        if "Finance" in rules.get(country, {}).get("departments_required", []):
            return True

    return False
