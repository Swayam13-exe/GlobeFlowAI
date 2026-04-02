from .models import EnvironmentState
from .rules import get_country_rules

def grade_easy(state: EnvironmentState) -> float:
    # Score: verified_docs / total_docs
    total = len(state.documents)
    if total == 0:
        return 0.0
    verified = sum(1 for doc in state.documents.values() if doc.status == "verified")
    return float(verified) / float(total)

def grade_medium(state: EnvironmentState) -> float:
    # 0.3 visa + 0.3 payroll + 0.4 compliance
    score = 0.0
    # Visa -> check passport and ep_application verified
    visa_docs_ok = True
    for doc in ["passport", "ep_application"]:
        if doc not in state.documents or state.documents[doc].status != "verified":
            visa_docs_ok = False
    if visa_docs_ok:
        score += 0.3
        
    if state.compliance.payroll:
        score += 0.3
        
    if state.compliance.pdpa and state.compliance.shadow_payroll:
        score += 0.4
        
    return min(1.0, score)

def grade_hard(state: EnvironmentState) -> float:
    # Score: 1.0 - (errors / total_checks)
    # total checks:
    # 1. passport verified
    # 2. visa_app verified
    # 3. tax_form verified
    # 4. uae_wps verified
    # 5. HR approved
    # 6. Legal approved
    # 7. Finance approved
    # 8. conflict resolution: in hard mode, tax is required by Germany but forbidden by UAE.
    # The correct case resolution is to reject (status="failed") OR resolve by bypassing tax.
    # Let's assess errors based on rule violations.
    errors = 0
    checks = 10
    
    docs_to_check = ["passport", "visa_application", "tax_form", "uae_wps_form"]
    for doc in docs_to_check:
        if state.documents.get(doc, None) is None or state.documents[doc].status != "verified":
            errors += 1
            
    if not state.departments.HR: errors += 1
    if not state.departments.Legal: errors += 1
    if not state.departments.Finance: errors += 1
    
    rules = get_country_rules(state.countries)
    # Germany requires tax, UAE forbids tax.
    # If tax_id is True, UAE is violated. If False, Germany is violated.
    # We will assume if "failed" state, they caught it.
    if state.status == "failed":
        # Conflict handled well
        pass
    else:
        # they must have done something. If tax_id is True, error for UAE. If False, error for Germany.
        errors += 1
        
    if state.status == "success":
        # Cannot be full success with conflict
        errors += 2

    # cap at 0 and 1
    raw_score = 1.0 - (float(errors) / float(checks))
    return max(0.0, min(1.0, raw_score))
