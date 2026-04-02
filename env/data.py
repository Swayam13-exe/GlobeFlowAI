from .models import EnvironmentState, Employee, DocumentState, Departments, Compliance

def get_base_state(task_level: str) -> EnvironmentState:
    documents = {}
    if task_level == "easy":
        countries = ["Germany"]
        documents = {
            "passport": DocumentState(status="missing", is_valid=True),
            "visa_application": DocumentState(status="missing", is_valid=True),
            "tax_form": DocumentState(status="submitted", is_valid=True)
        }
    elif task_level == "medium":
        countries = ["Singapore"]
        documents = {
            "passport": DocumentState(status="submitted", is_valid=True),
            "ep_application": DocumentState(status="missing", is_valid=True),
            "pdpa_form": DocumentState(status="missing", is_valid=True)
        }
    elif task_level == "hard":
        countries = ["Germany", "UAE"]
        documents = {
            "passport": DocumentState(status="submitted", is_valid=True),
            "visa_application": DocumentState(status="missing", is_valid=True),
            "tax_form": DocumentState(status="missing", is_valid=True),
            "uae_wps_form": DocumentState(status="submitted", is_valid=True)
        }
    else:
        raise ValueError(f"Unknown task level: {task_level}")
        
    return EnvironmentState(
        case_id=f"CASE_{task_level.upper()}_001",
        employee=Employee(role="Software Engineer", has_dependents=False),
        countries=countries,
        documents=documents,
        departments=Departments(HR=False, Legal=False, Finance=False),
        compliance=Compliance(tax_id=False, payroll=False, pdpa=False, shadow_payroll=False),
        deadline_days=30,
        previous_actions=[],
        progress=0.0,
        status="in_progress"
    )
