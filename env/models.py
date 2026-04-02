from pydantic import BaseModel, Field
from typing import Dict, List, Literal

class Employee(BaseModel):
    role: str
    has_dependents: bool

class DocumentState(BaseModel):
    status: Literal["missing", "submitted", "verified", "rejected"]
    is_valid: bool

class Departments(BaseModel):
    HR: bool = False
    Legal: bool = False
    Finance: bool = False

class Compliance(BaseModel):
    tax_id: bool = False
    payroll: bool = False
    pdpa: bool = False
    shadow_payroll: bool = False

class EnvironmentState(BaseModel):
    case_id: str
    employee: Employee
    countries: List[str]
    documents: Dict[str, DocumentState]
    departments: Departments
    compliance: Compliance
    deadline_days: int
    previous_actions: List[str] = Field(default_factory=list)
    progress: float = 0.0
    status: Literal["in_progress", "success", "failed"] = "in_progress"

class Action(BaseModel):
    action_type: Literal[
        "request_document", 
        "verify_document", 
        "approve_hr", 
        "approve_legal", 
        "approve_finance", 
        "set_payroll", 
        "set_tax_id", 
        "set_shadow_payroll", 
        "set_pdpa", 
        "finalize_case"
    ]
    target: str
