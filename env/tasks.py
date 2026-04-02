"""
env/tasks.py
============
Task fixtures for openenv-workforce.

TASKS is a dict mapping task_name → initial state dict.
environment.py deep-copies the relevant entry on reset() so the fixture
is never mutated between episodes.

Task summary:
  easy:   India → Germany, Engineer, no dependents, 10 days, linear path
  medium: India → Singapore, Manager with dependents, more compliance overhead
  hard:   India → Germany + UAE simultaneously, Director, is_valid traps,
          UAE no-tax rule as the critical pitfall

State dict schema matches WorkforceState in env/models.py:
  case_id          str
  employee         { role, has_dependents }
  countries        list[str]                  ← 1 for easy/medium, 2 for hard
  documents        { doc_name: { status, is_valid } }
  departments      { HR, Legal, Finance }
  compliance       { tax_id, payroll, pdpa, shadow_payroll }
  deadline_days    int
  previous_actions list[str]
  progress         float
  status           str

Author: Team AI Kalesh
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# TASKS registry
# ---------------------------------------------------------------------------

TASKS: dict[str, dict] = {

    # =========================================================================
    # TASK 1 — EASY
    # Scenario: Relocate an Engineer from India to Germany.
    # Expected competent score: 0.70 – 1.00
    #
    # What the agent must do:
    #   1. request_document + verify_document for all 4 Germany docs
    #   2. approve_hr
    #   3. set_tax_id:Germany
    #   4. set_payroll:Germany
    #   5. finalize_case:all
    #
    # All documents start is_valid=True for easy task (no traps).
    # Legal approval NOT required (Germany easy skip) — HR only.
    # deadline_days=10 is generous — agent should finish in ≤8 steps.
    # =========================================================================
    "easy": {
        "case_id": "CASE-001-EASY",
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
                "is_valid": True,   # EU Blue Card valid for Engineer salary
            },
            "employment_letter": {
                "status":   "missing",
                "is_valid": True,
            },
            "degree_certificate": {
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
        "deadline_days":    20,
        "previous_actions": [],
        "progress":         0.0,
        "status":           "in_progress",
    },

    # =========================================================================
    # TASK 2 — MEDIUM
    # Scenario: Relocate a Manager with dependents from India to Singapore.
    # Expected competent score: 0.40 – 0.80
    #
    # What the agent must do:
    #   1. request_document + verify_document for 3 Singapore docs
    #   2. approve_hr
    #   3. approve_legal          (requires all docs verified first)
    #   4. set_tax_id:Singapore
    #   5. set_payroll:Singapore
    #   6. set_shadow_payroll:Singapore
    #   7. set_pdpa:Singapore
    #   8. approve_finance        (requires Legal approved first)
    #   9. finalize_case:all
    #
    # Traps:
    #   - Degree certificate NOT required for Singapore (wrong_action if requested)
    #   - degree_certificate.is_valid=False introduces a trap for careless agents
    #   - PDPA and shadow payroll must be set before finalizing
    #   - Legal must come before Finance
    # =========================================================================
    "medium": {
        "case_id": "CASE-002-MEDIUM",
        "employee": {
            "role":           "Manager",
            "has_dependents": True,   # dependents increase compliance burden
        },
        "countries": ["Singapore"],
        "documents": {
            "passport": {
                "status":   "missing",
                "is_valid": True,
            },
            "visa": {
                "status":   "missing",
                "is_valid": True,   # Employment Pass valid for Manager
            },
            "employment_letter": {
                "status":   "missing",
                "is_valid": True,
            },
            "degree_certificate": {
                "status":   "missing",
                "is_valid": False,  # Trap: degree cert not required for SG,
                                    # and is_valid=False so verify will reject it
                                    # if the agent wastes a step on it
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
        "deadline_days":    25,   # more steps needed for Singapore complexity
        "previous_actions": [],
        "progress":         0.0,
        "status":           "in_progress",
    },

    # =========================================================================
    # TASK 3 — HARD
    # Scenario: Relocate a Director from India to Germany AND UAE simultaneously.
    # Expected competent score: 0.20 – 0.60
    #
    # What the agent must do:
    #   1. request_document + verify_document for Germany docs (4 docs)
    #   2. request_document + verify_document for UAE docs (3 docs — shared with DE)
    #   3. approve_hr
    #   4. approve_legal          (requires all docs verified)
    #   5. set_tax_id:Germany     ← DO THIS
    #   6. set_payroll:Germany
    #   7. set_payroll:UAE
    #                             ← DO NOT set_tax_id:UAE (UAE has no income tax!)
    #   8. approve_finance
    #   9. finalize_case:all
    #
    # Critical traps:
    #   - UAE has NO income tax → set_tax_id:UAE = rule_violation (-0.3)
    #   - Some docs have is_valid=False to test prerequisite handling
    #   - passport.is_valid=False initially (simulate expired passport scenario)
    #     Agent must request → see rejection → this is a simplified sim so
    #     after request the env sets is_valid based on rules, making passport valid
    #     (see _handle_request_document in environment.py)
    #   - Finance required (multi-country OR UAE/SG rules)
    #   - Deadline is tight (20 days for many more required steps)
    # =========================================================================
    "hard": {
        "case_id": "CASE-003-HARD",
        "employee": {
            "role":           "Director",
            "has_dependents": True,
        },
        "countries": ["Germany", "UAE"],
        "documents": {
            "passport": {
                "status":   "missing",
                "is_valid": True,   # shared doc — valid for both countries
            },
            "visa": {
                "status":   "missing",
                "is_valid": True,   # EU Blue Card (Germany) + Employment Visa (UAE)
                                    # validity resolved by country rules in env
            },
            "employment_letter": {
                "status":   "missing",
                "is_valid": True,
            },
            "degree_certificate": {
                "status":   "missing",
                "is_valid": True,   # Required for Germany (EU Blue Card)
            },
        },
        "departments": {
            "HR":      False,
            "Legal":   False,
            "Finance": False,   # Finance required — multi-country case
        },
        "compliance": {
            "tax_id":         False,
            "payroll":        False,
            "pdpa":           False,   # NOT required for Germany or UAE
            "shadow_payroll": False,   # NOT required for Germany or UAE
        },
        "deadline_days":    30,   # multi-country needs at minimum ~14 steps
        "previous_actions": [],
        "progress":         0.0,
        "status":           "in_progress",
    },
}


# ---------------------------------------------------------------------------
# Task metadata (used by ResetResult.task_info in models.py)
# ---------------------------------------------------------------------------

TASK_INFO: dict[str, dict] = {
    "easy": {
        "name":                 "easy",
        "description":          "Relocate an Engineer from India to Germany (EU Blue Card pathway). Single country, no dependents, all documents valid.",
        "countries":            ["Germany"],
        "max_steps":            25,
        "expected_score_range": (0.70, 1.00),
        "departments_required": ["HR", "Legal"],
        "key_rules": [
            "Germany requires EU Blue Card visa for skilled workers",
            "Tax ID registration required with Bundeszentralamt für Steuern",
            "GDPR applies to all employee data processing",
        ],
    },
    "medium": {
        "name":                 "medium",
        "description":          "Relocate a Manager with dependents from India to Singapore (Employment Pass). Requires PDPA consent and shadow payroll.",
        "countries":            ["Singapore"],
        "max_steps":            25,
        "expected_score_range": (0.40, 0.80),
        "departments_required": ["HR", "Legal", "Finance"],
        "key_rules": [
            "Singapore requires Employment Pass for professionals (NOT S Pass)",
            "PDPA consent must be collected before any data processing",
            "Shadow payroll mandatory for home-country tax tracking",
            "Finance approval required (comes after Legal)",
        ],
    },
    "hard": {
        "name":                 "hard",
        "description":          "Simultaneous relocation of a Director from India to Germany AND UAE. Multi-country compliance. Key trap: UAE has NO income tax.",
        "countries":            ["Germany", "UAE"],
        "max_steps":            25,
        "expected_score_range": (0.20, 0.60),
        "departments_required": ["HR", "Legal", "Finance"],
        "key_rules": [
            "UAE has NO income tax — set_tax_id:UAE is a rule violation (-0.3)",
            "Germany requires tax ID registration",
            "Both countries require payroll configuration",
            "Finance approval required for multi-country cases",
            "Documents must be verified for BOTH countries before Legal approval",
        ],
    },
}
