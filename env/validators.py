"""
env/validators.py
=================
Pure validation functions for openenv-workforce.

All functions:
  - Are pure — no side effects, no state mutation
  - Return (is_valid: bool, reason: str)
  - Are called by environment.py BEFORE any state is mutated
  - Implement the three types of checks the master prompt requires:

    validate_document(state, doc_name)
      → Check is_valid flag and submission status before verifying

    validate_department_prerequisites(state, dept)
      → Enforce HR → Legal → Finance ordering + doc prerequisites

    validate_compliance_action(state, action_type, country)
      → Enforce country-specific rules (UAE no-tax, Singapore-only shadow payroll)

Used by environment.py action handlers:
    _handle_verify_document        → validate_document
    _handle_approve_department     → validate_department_prerequisites
    _handle_set_tax_id             → validate_compliance_action
    _handle_set_shadow_payroll     → validate_compliance_action
    _handle_set_pdpa               → validate_compliance_action

Author: Team AI Kalesh
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# validate_document
# ---------------------------------------------------------------------------


def validate_document(
    state: dict[str, Any],
    doc_name: str,
) -> tuple[bool, str]:
    """
    Validate whether a document can be successfully verified.

    This is called by _handle_verify_document AFTER the document has been
    confirmed to exist and be in 'submitted' status.

    Validation checks (in order):
      1. Document must exist in state
      2. Document must be in 'submitted' status (can't verify 'missing' or already done)
      3. is_valid ground-truth flag must be True (set at task load time,
         reflects whether the submitted document is actually valid)
      4. For visa documents: country rules must require a visa

    Args:
        state:    Current episode state dict.
        doc_name: Name of the document to validate.

    Returns:
        (True, "ok") if document passes validation.
        (False, reason_string) if it fails.
    """
    docs = state.get("documents", {})
    doc  = docs.get(doc_name)

    # ── 1. Document must exist ───────────────────────────────────────────────
    if doc is None:
        return False, f"Document '{doc_name}' does not exist in this case"

    # ── 2. Status must be 'submitted' ───────────────────────────────────────
    status = doc.get("status", "missing")
    if status == "missing":
        return (
            False,
            f"Document '{doc_name}' has not been submitted yet. "
            f"Call request_document first.",
        )
    if status == "verified":
        return False, f"Document '{doc_name}' is already verified"
    if status == "rejected":
        return (
            False,
            f"Document '{doc_name}' was previously rejected. "
            f"Re-request and resubmit before verifying again.",
        )

    # ── 3. Ground-truth validity check ──────────────────────────────────────
    if not doc.get("is_valid", False):
        return (
            False,
            f"Document '{doc_name}' failed validation — "
            f"the submitted document is not valid for this relocation case.",
        )

    # ── 4. Visa-specific: country must actually require a visa ──────────────
    if doc_name == "visa":
        countries = state.get("countries", [])
        from env.rules_engine import _load_country_rules
        rules = _load_country_rules()
        any_needs_visa = any(
            rules.get(c, {}).get("requires_visa", False) for c in countries
        )
        if not any_needs_visa:
            return (
                False,
                "Visa is not required for any country in this case.",
            )

    return True, "ok"


# ---------------------------------------------------------------------------
# validate_department_prerequisites
# ---------------------------------------------------------------------------


def validate_department_prerequisites(
    state: dict[str, Any],
    dept: str,
) -> tuple[bool, str]:
    """
    Validate that all prerequisites for approving a department are met.

    Ordering constraints enforced:
      HR:      No prerequisites — can always be approved.
      Legal:   All required documents for ALL countries must be verified.
      Finance: Legal must already be approved.

    Args:
        state: Current episode state dict.
        dept:  Department name — "HR" | "Legal" | "Finance".

    Returns:
        (True, "ok") if all prerequisites are met.
        (False, reason_string) if any prerequisite is unmet.
    """
    countries   = state.get("countries", [])
    departments = state.get("departments", {})
    documents   = state.get("documents", {})

    if dept == "HR":
        # HR has no prerequisites
        return True, "ok"

    elif dept == "Legal":
        # All required documents for all countries must be verified
        from env.rules_engine import get_required_documents
        for country in countries:
            required_docs = get_required_documents(country)
            for doc_name in required_docs:
                doc_status = documents.get(doc_name, {}).get("status", "missing")
                if doc_status != "verified":
                    return (
                        False,
                        f"Legal approval requires all documents to be verified first. "
                        f"Document '{doc_name}' for {country} has status '{doc_status}'. "
                        f"Verify all documents before approving Legal.",
                    )
        return True, "ok"

    elif dept == "Finance":
        # Legal must be approved first
        if not departments.get("Legal", False):
            return (
                False,
                "Finance approval requires Legal approval first. "
                "Approve Legal before approving Finance.",
            )
        return True, "ok"

    else:
        return False, f"Unknown department '{dept}'"


# ---------------------------------------------------------------------------
# validate_compliance_action
# ---------------------------------------------------------------------------


def validate_compliance_action(
    state: dict[str, Any],
    action_type: str,
    country: str,
) -> tuple[bool, str]:
    """
    Validate a compliance-setting action against country-specific rules.

    Enforces critical domain rules:

      set_tax_id:
        ❌ UAE has NO income tax — calling set_tax_id for UAE is ALWAYS a
           rule_violation regardless of any other state. This is the key
           trap in the hard task (India → Germany + UAE).

      set_shadow_payroll:
        ❌ Only Singapore requires shadow payroll. Calling this for any
           other country is a rule_violation.

      set_pdpa:
        ❌ Only Singapore requires PDPA consent. Calling this for any
           other country is a rule_violation.

      set_payroll:
        ✅ Required for all countries — validated separately in environment.py.
           No special rule check needed here.

    Args:
        state:       Current episode state dict.
        action_type: One of "set_tax_id" | "set_shadow_payroll" | "set_pdpa".
        country:     Target country for the compliance action.

    Returns:
        (True, "ok") if the action is compliant with country rules.
        (False, reason_string) if it violates a rule.
    """
    from env.rules_engine import _load_country_rules
    rules = _load_country_rules()
    country_rules = rules.get(country, {})

    if action_type == "set_tax_id":
        # ── CRITICAL: UAE no-income-tax rule ────────────────────────────────
        if not country_rules.get("has_income_tax", True):
            return (
                False,
                f"RULE VIOLATION: {country} has no income tax. "
                f"Tax ID registration must NOT be called for {country}. "
                f"This action incurs a -0.3 penalty. "
                f"Check country rules before registering tax IDs.",
            )

        if not country_rules.get("requires_tax_id", False):
            return (
                False,
                f"Tax ID is not required for {country}.",
            )

        return True, "ok"

    elif action_type == "set_shadow_payroll":
        # ── Singapore-only rule ──────────────────────────────────────────────
        if not country_rules.get("requires_shadow_payroll", False):
            return (
                False,
                f"Shadow payroll is not required for {country}. "
                f"Only Singapore requires shadow payroll. "
                f"This action is a rule_violation for {country}.",
            )
        return True, "ok"

    elif action_type == "set_pdpa":
        # ── Singapore-only rule ──────────────────────────────────────────────
        if not country_rules.get("requires_pdpa", False):
            return (
                False,
                f"PDPA consent is not required for {country}. "
                f"Only Singapore requires PDPA. "
                f"This action is a rule_violation for {country}.",
            )
        return True, "ok"

    # Unknown compliance action — pass through (environment.py will handle)
    return True, "ok"
