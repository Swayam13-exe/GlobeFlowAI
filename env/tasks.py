"""
env/tasks.py
============
Task fixtures for openenv-workforce.

TASKS is a dict mapping task_name → initial state dict.
environment.py deep-copies the relevant entry on reset() so the fixture
is never mutated between episodes.

Task summary:
  easy:   India → Germany, Engineer, no dependents, 20 days
          Required: 4 docs + HR approval + tax_id + payroll
          
  medium: India → Singapore, Manager with dependents, 25 days
          Required: 3 docs + HR + Legal + payroll + pdpa + shadow_payroll
          
  hard:   India → Germany + UAE simultaneously, Director, 30 days
          Required: 4 docs + HR + Legal + Finance + tax_id(DE only) + payroll(both)
          KEY TRAP: UAE has no income tax — set_tax_id:UAE = rule_violation

State dict schema matches WorkforceState in env/models.py exactly.

Author: Team AI Kalesh
"""

from __future__ import annotations


TASKS: dict[str, dict] = {

    # =========================================================================
    # TASK 1 — EASY
    # India → Germany, Engineer, no dependents
    #
    # Optimal sequence (~9 steps):
    #   request_document x4 → verify_document x4 → approve_hr →
    #   set_tax_id → set_payroll → finalize_case
    #
    # All documents is_valid=True — no traps.
    # Only HR required (Legal/Finance not required for easy).
    # =========================================================================
    "easy": {
        "case_id":   "CASE-001-EASY",
        "task_name": "easy",                          # FIX 6: added task_name
        "employee": {
            "role":           "Engineer",
            "has_dependents": False,
        },
        "countries": ["Germany"],
        "documents": {
            "passport": {
                "status":   "missing",
                "is_valid": True,
            },
            "visa": {
                "status":   "missing",
                "is_valid": True,
            },
            "employment_letter": {
                "status":   "missing",
                "is_valid": True,
            },
            "work_permit": {
                "status":   "missing",
                "is_valid": True,
            },
        },
        "departments": {
            "HR":      False,
            "Legal":   False,
            "Finance": False,
        },
        "compliance": {
            "tax_id":         False,
            "payroll":        False,
            "pdpa":           False,
            "shadow_payroll": False,
        },
        "conflicts": [],                              # FIX 4: always present
        "deadline_days":      20,
        "previous_actions":   [],
        "progress":           0.0,
        "status":             "in_progress",
        # FIX 1: required lists — used by validators, graders, reward
        "required_departments": ["HR"],
        "required_compliance":  ["tax_id", "payroll"],
    },

    # =========================================================================
    # TASK 2 — MEDIUM
    # India → Singapore, Manager with dependents
    #
    # Optimal sequence (~12 steps):
    #   request_document x3 → verify_document x3 → approve_hr →
    #   approve_legal → set_payroll → set_pdpa → set_shadow_payroll →
    #   finalize_case
    #
    # Note: degree_certificate NOT included in docs — agent only sees the
    # 3 documents relevant to Singapore. No invalid-doc trap.
    # Singapore does NOT require tax_id (no income tax for foreigners).
    # Finance NOT required for Singapore medium task.
    # =========================================================================
    "medium": {
        "case_id":   "CASE-002-MEDIUM",
        "task_name": "medium",                        # FIX 6
        "employee": {
            "role":           "Manager",
            "has_dependents": True,
        },
        "countries": ["Singapore"],
        "documents": {
            "passport": {
                "status":   "missing",
                "is_valid": True,
            },
            "visa": {
                "status":   "missing",
                "is_valid": True,
            },
            "employment_letter": {
                "status":   "missing",
                "is_valid": True,
            },
            # FIX 3: degree_certificate removed — it's not required for
            # Singapore and having is_valid=False permanently blocks Legal.
            # Trap is now logical (Singapore rules) not a broken dead-end.
        },
        "departments": {
            "HR":      False,
            "Legal":   False,
            "Finance": False,
        },
        "compliance": {
            "tax_id":         False,
            "payroll":        False,
            "pdpa":           False,
            "shadow_payroll": False,
        },
        "conflicts": [],                              # FIX 4
        "deadline_days":      25,
        "previous_actions":   [],
        "progress":           0.0,
        "status":             "in_progress",
        # FIX 1 + FIX 5: Singapore medium — HR + Legal, no Finance
        "required_departments": ["HR", "Legal"],
        "required_compliance":  ["payroll", "pdpa", "shadow_payroll"],
    },

    # =========================================================================
    # TASK 3 — HARD
    # India → Germany + UAE simultaneously, Director with dependents
    #
    # Optimal sequence (~16 steps):
    #   request_document x4 → verify_document x4 → approve_hr →
    #   approve_legal → set_tax_id (Germany ONLY) → set_payroll →
    #   resolve_conflict → approve_finance → finalize_case
    #
    # KEY TRAPS:
    #   1. UAE has NO income tax → set_tax_id:UAE = -0.3 rule_violation
    #      The conflict record makes this explicit — agent must resolve it.
    #   2. Finance requires Legal + all conflicts resolved
    #   3. Tight deadline (30 days) for ~16 required steps
    #   4. Both countries' docs must be verified before Legal
    #
    # Documents: shared docs cover both countries (passport, visa,
    # employment_letter, work_permit). UAE uses residence_permit instead of
    # work_permit but for simplicity this task uses a combined doc set.
    # =========================================================================
    "hard": {
        "case_id":   "CASE-003-HARD",
        "task_name": "hard",                          # FIX 6
        "employee": {
            "role":           "Director",
            "has_dependents": True,
        },
        "countries": ["Germany", "UAE"],
        "documents": {
            "passport": {
                "status":   "missing",
                "is_valid": True,
            },
            "visa": {
                "status":   "missing",
                "is_valid": True,
            },
            "employment_letter": {
                "status":   "missing",
                "is_valid": True,
            },
            "work_permit": {
                "status":   "missing",
                "is_valid": True,
            },
        },
        "departments": {
            "HR":      False,
            "Legal":   False,
            "Finance": False,
        },
        "compliance": {
            "tax_id":         False,
            "payroll":        False,
            "pdpa":           False,
            "shadow_payroll": False,
        },
        # FIX 4: conflicts key — the Germany+UAE tax conflict
        # Agent must call resolve_conflict before Finance can approve
        "conflicts": [
            {
                "countries": ["Germany", "UAE"],
                "rule": (
                    "tax_conflict: Germany requires tax_id registration "
                    "but UAE has no income tax. "
                    "Register tax_id for Germany only. "
                    "Do NOT call set_tax_id for UAE."
                ),
                "resolved": False,
            }
        ],
        "deadline_days":      30,
        "previous_actions":   [],
        "progress":           0.0,
        "status":             "in_progress",
        # FIX 1: all three departments + tax_id + payroll required
        "required_departments": ["HR", "Legal", "Finance"],
        "required_compliance":  ["tax_id", "payroll"],
    },
}


# ---------------------------------------------------------------------------
# Task metadata — used by ResetResult.task_info and TASK_INFO lookups
# ---------------------------------------------------------------------------

TASK_INFO: dict[str, dict] = {
    "easy": {
        "name":        "easy",
        "description": (
            "Relocate an Engineer from India to Germany. "
            "Single country, no dependents, all documents valid. "
            "Requires: 4 docs verified + HR approval + tax_id + payroll."
        ),
        "countries":            ["Germany"],
        "max_steps":            25,
        "expected_score_range": (0.70, 1.00),
        # FIX 2: consistent with required_departments in task dict
        "departments_required": ["HR"],
        "compliance_required":  ["tax_id", "payroll"],
        "key_rules": [
            "Germany requires visa and work_permit",
            "Tax ID registration required for Germany",
            "Only HR approval required for easy task",
        ],
    },
    "medium": {
        "name":        "medium",
        "description": (
            "Relocate a Manager with dependents from India to Singapore. "
            "Requires Employment Pass, PDPA consent, shadow payroll. "
            "HR + Legal approval required."
        ),
        "countries":            ["Singapore"],
        "max_steps":            25,
        "expected_score_range": (0.40, 0.80),
        # FIX 5: HR + Legal only for Singapore medium
        "departments_required": ["HR", "Legal"],
        "compliance_required":  ["payroll", "pdpa", "shadow_payroll"],
        "key_rules": [
            "Singapore requires PDPA consent before data processing",
            "Shadow payroll mandatory for home-country tax tracking",
            "Singapore does NOT require tax_id for foreign workers",
            "Legal must approve before finalize",
        ],
    },
    "hard": {
        "name":        "hard",
        "description": (
            "Simultaneous relocation of a Director from India to Germany + UAE. "
            "Multi-country compliance with conflicting tax rules. "
            "KEY TRAP: UAE has no income tax — set_tax_id:UAE is a rule violation."
        ),
        "countries":            ["Germany", "UAE"],
        "max_steps":            25,
        "expected_score_range": (0.20, 0.60),
        "departments_required": ["HR", "Legal", "Finance"],
        "compliance_required":  ["tax_id", "payroll"],
        "key_rules": [
            "UAE has NO income tax — set_tax_id:UAE = -0.3 penalty",
            "Germany requires tax_id — call set_tax_id for Germany only",
            "Both countries require payroll configuration",
            "Finance requires Legal approval AND all conflicts resolved",
            "resolve_conflict must be called before approve_finance",
        ],
    },
}