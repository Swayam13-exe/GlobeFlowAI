"""
env/models.py
=============
All Pydantic v2 models for openenv-workforce.

Schema hierarchy:
  EmployeeRecord          — who is being relocated
  DocumentRecord          — a single document's status + validity
  DepartmentStatus        — HR / Legal / Finance approval flags
  ComplianceStatus        — tax, payroll, PDPA, shadow payroll flags
  WorkforceState          — complete episode state (single source of truth)
  Action                  — one agent action: {action_type, target}
  Observation             — what the agent sees after each step
  Reward                  — per-step reward with reason
  StepResult              — full return value of step()
  ResetResult             — return value of reset()
  TaskInfo                — metadata for a single task
  EpisodeSummary          — end-of-episode summary with final grader score

All models use Pydantic v2 syntax exclusively.
Never use @validator — use @field_validator or @model_validator instead.

Author: Team AI Kalesh
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations (as Literal types — avoids Enum overhead, stays JSON-friendly)
# ---------------------------------------------------------------------------

DocumentStatus = Literal["missing", "submitted", "verified", "rejected"]
EpisodeStatus  = Literal["in_progress", "success", "failed"]
TaskName       = Literal["easy", "medium", "hard"]
CountryName    = Literal["Germany", "Singapore", "UAE"]
RoleName       = Literal["Engineer", "Manager", "Director", "Analyst"]

DepartmentName = Literal["HR", "Legal", "Finance"]
ActionType     = Literal[
    "request_document",
    "verify_document",
    "approve_hr",
    "approve_legal",
    "approve_finance",
    "set_payroll",
    "set_tax_id",
    "set_shadow_payroll",
    "set_pdpa",
    "finalize_case",
]

DocumentName = Literal[
    "passport",
    "visa",
    "employment_letter",
    "degree_certificate",
]


# ---------------------------------------------------------------------------
# Sub-models — building blocks for WorkforceState
# ---------------------------------------------------------------------------


class EmployeeRecord(BaseModel):
    """
    Describes the employee being relocated.

    Fields:
        role:            Job title / seniority level.
        has_dependents:  True if spouse or children are moving too.
                         Affects visa type eligibility and case complexity.
    """

    role: RoleName = Field(
        ...,
        description="Employee job role — affects visa eligibility and complexity",
        examples=["Engineer", "Manager"],
    )
    has_dependents: bool = Field(
        default=False,
        description="Whether dependents (spouse/children) are included in the relocation",
    )

    model_config = {"frozen": True}   # immutable after construction


class DocumentRecord(BaseModel):
    """
    Tracks a single relocation document's lifecycle.

    Status transitions:
        missing → submitted → verified
                           → rejected

    Fields:
        status:   Current lifecycle stage.
        is_valid: Ground-truth validity (set at task load time).
                  An invalid document will always be rejected on verify_document.
                  The agent does not see this field directly — it must infer
                  validity from the verify_document result.
    """

    status: DocumentStatus = Field(
        default="missing",
        description="Current document lifecycle stage",
    )
    is_valid: bool = Field(
        default=False,
        description=(
            "Simulated ground-truth validity. "
            "Determines verify_document outcome. "
            "Not directly observable by the agent."
        ),
    )

    @field_validator("status")
    @classmethod
    def status_must_be_valid(cls, v: str) -> str:
        allowed = {"missing", "submitted", "verified", "rejected"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}, got '{v}'")
        return v


class DepartmentStatus(BaseModel):
    """
    Tracks approval status across the three internal departments.

    Approval ordering constraint (enforced by validators.py):
        HR     → no prerequisites
        Legal  → all required documents must be verified first
        Finance → Legal must be approved first

    Each field is True once that department has approved the case.
    """

    HR:      bool = Field(default=False, description="HR department approval")
    Legal:   bool = Field(default=False, description="Legal department approval")
    Finance: bool = Field(default=False, description="Finance department approval")

    @model_validator(mode="after")
    def finance_requires_legal(self) -> "DepartmentStatus":
        """
        Finance approval without Legal approval is an invalid state.
        This should never occur via normal step() calls (validators prevent it),
        but we enforce it at the model level as a safety net.
        """
        if self.Finance and not self.Legal:
            raise ValueError(
                "Finance cannot be approved before Legal. "
                "Invalid state — check your action sequence."
            )
        return self

    def approved_count(self) -> int:
        """Return how many departments have approved."""
        return sum([self.HR, self.Legal, self.Finance])

    def all_approved(self) -> bool:
        """True when all three departments have approved."""
        return self.HR and self.Legal and self.Finance


class ComplianceStatus(BaseModel):
    """
    Tracks country-specific compliance items.

    Which fields are required depends on destination country:
        Germany:   tax_id + payroll
        Singapore: tax_id + payroll + pdpa + shadow_payroll
        UAE:       payroll only  (no income tax — tax_id must NOT be set)

    Fields:
        tax_id:         Tax registration with host country authority.
        payroll:        Host-country payroll configured.
        pdpa:           Singapore Personal Data Protection Act consent.
        shadow_payroll: Singapore shadow payroll for home-country tax tracking.
    """

    tax_id:         bool = Field(default=False, description="Tax ID registered with host country")
    payroll:        bool = Field(default=False, description="Host country payroll configured")
    pdpa:           bool = Field(default=False, description="Singapore PDPA consent collected")
    shadow_payroll: bool = Field(default=False, description="Shadow payroll enabled (Singapore only)")

    def completed_count(self) -> int:
        """Return number of compliance items that are True."""
        return sum([self.tax_id, self.payroll, self.pdpa, self.shadow_payroll])


# ---------------------------------------------------------------------------
# Master state model
# ---------------------------------------------------------------------------


class WorkforceState(BaseModel):
    """
    The complete episode state — single source of truth for the environment.

    This is returned by state() and embedded in every Observation.
    All fields map directly to the internal state dict in environment.py.

    Fields:
        case_id:          Unique identifier for this relocation case.
        employee:         The employee being relocated.
        countries:        Destination country/countries (1 for easy/medium, 2 for hard).
        documents:        Map of document_name → DocumentRecord.
        departments:      HR / Legal / Finance approval flags.
        compliance:       Country-specific compliance completion flags.
        deadline_days:    Days remaining before episode auto-fails.
                          Decrements by 1 each step().
        previous_actions: History of "action_type:target" strings.
                          Used to detect and penalise repeated actions.
        progress:         Float [0.0, 1.0] — fraction of checklist completed.
                          Updated after every step.
        status:           Episode lifecycle stage.
    """

    case_id:           str                          = Field(..., description="Unique case identifier")
    employee:          EmployeeRecord               = Field(..., description="Employee being relocated")
    countries:         list[str]                    = Field(..., min_length=1, max_length=2, description="Destination countries")
    documents:         dict[str, DocumentRecord]    = Field(default_factory=dict, description="Document name → status + validity")
    departments:       DepartmentStatus             = Field(default_factory=DepartmentStatus, description="Department approval flags")
    compliance:        ComplianceStatus             = Field(default_factory=ComplianceStatus, description="Compliance completion flags")
    deadline_days:     int                          = Field(default=5, ge=0, description="Steps remaining before auto-fail")
    previous_actions:  list[str]                    = Field(default_factory=list, description="Action history for repeat detection")
    progress:          float                        = Field(default=0.0, ge=0.0, le=1.0, description="Checklist completion fraction")
    status:            EpisodeStatus                = Field(default="in_progress", description="Episode lifecycle status")

    @field_validator("countries")
    @classmethod
    def countries_must_be_valid(cls, v: list[str]) -> list[str]:
        valid = {"Germany", "Singapore", "UAE"}
        for c in v:
            if c not in valid:
                raise ValueError(f"Country '{c}' not supported. Valid: {valid}")
        return v

    @field_validator("progress")
    @classmethod
    def progress_in_range(cls, v: float) -> float:
        return round(max(0.0, min(1.0, v)), 4)

    @field_validator("deadline_days")
    @classmethod
    def deadline_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("deadline_days cannot be negative")
        return v

    def is_done(self) -> bool:
        """True when the episode has ended (success or failure)."""
        return self.status in {"success", "failed"}

    def get_verified_documents(self) -> list[str]:
        """Return names of all verified documents."""
        return [name for name, doc in self.documents.items() if doc.status == "verified"]

    def get_missing_documents(self) -> list[str]:
        """Return names of all documents still at 'missing' status."""
        return [name for name, doc in self.documents.items() if doc.status == "missing"]

    def get_rejected_documents(self) -> list[str]:
        """Return names of all rejected documents."""
        return [name for name, doc in self.documents.items() if doc.status == "rejected"]


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------


class Action(BaseModel):
    """
    One agent action submitted to step().

    Format:
        { "action_type": "<type>", "target": "<target>" }

    Valid action_type values and their expected targets:
        request_document    → passport | visa | employment_letter | degree_certificate
        verify_document     → passport | visa | employment_letter | degree_certificate
        approve_hr          → HR
        approve_legal       → Legal
        approve_finance     → Finance
        set_payroll         → Germany | Singapore | UAE
        set_tax_id          → Germany | Singapore  (NOT UAE — no income tax)
        set_shadow_payroll  → Singapore
        set_pdpa            → Singapore
        finalize_case       → all

    Invalid action_type or target returns reward=-0.2, state unchanged.
    """

    action_type: str = Field(
        ...,
        description="The type of action to perform",
        examples=["verify_document", "approve_hr", "set_tax_id"],
    )
    target: str = Field(
        ...,
        description="The subject of the action — document name, department, or country",
        examples=["passport", "HR", "Germany"],
    )

    @field_validator("action_type")
    @classmethod
    def action_type_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("action_type cannot be empty")
        return v.strip().lower()

    @field_validator("target")
    @classmethod
    def target_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("target cannot be empty")
        return v.strip()

    def to_key(self) -> str:
        """Return 'action_type:target' string used for repeat detection."""
        return f"{self.action_type}:{self.target}"


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------


class Reward(BaseModel):
    """
    Per-step reward returned alongside the observation.

    Fields:
        value:  Float in [-1.0, 1.0] for per-step rewards.
                Cumulative episode score is clamped to [0.0, 1.0].
        reason: Human-readable explanation of why this reward was given.
                Useful for agent debugging and trajectory analysis.
    """

    value:  float = Field(..., description="Reward value clamped to [-1.0, 1.0]")
    reason: str   = Field(..., description="Human-readable reward explanation")

    @field_validator("value")
    @classmethod
    def clamp_reward(cls, v: float) -> float:
        """Hard clamp — reward must always be in [-1.0, 1.0]."""
        return round(max(-1.0, min(1.0, v)), 4)


# ---------------------------------------------------------------------------
# Observation model — what the agent sees after every step
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """
    The full observation returned to the agent by reset() and step().

    Fields:
        state:               Complete current WorkforceState.
        available_actions:   List of "action_type:target" strings the agent
                             can usefully take right now. This is a hint —
                             not a hard restriction. Actions outside this list
                             are still accepted but likely penalised.
        current_blockers:    Human-readable list of reasons the case cannot
                             be finalized yet. Empty = finalize_case is valid.
        last_action_result:  String describing the result of the last action:
                             "success" | "invalid_action" | "rule_violation" |
                             "prereq_violated" | "wrong_action" | "repeated_action"
        last_action_error:   Detailed error message if last action failed, else None.
        steps_taken:         Number of step() calls made so far this episode.
        done:                True when the episode has ended.
    """

    state:               WorkforceState = Field(..., description="Complete current episode state")
    available_actions:   list[str]      = Field(default_factory=list, description="Suggested valid actions as 'action_type:target' strings")
    current_blockers:    list[str]      = Field(default_factory=list, description="Reasons case cannot be finalized yet")
    last_action_result:  str            = Field(default="", description="Result code of the last action")
    last_action_error:   str | None     = Field(default=None, description="Error detail if last action failed")
    steps_taken:         int            = Field(default=0, ge=0, description="Steps taken so far this episode")
    done:                bool           = Field(default=False, description="True when episode has ended")

    def is_success(self) -> bool:
        """True when the episode ended with status=success."""
        return self.done and self.state.status == "success"

    def is_failed(self) -> bool:
        """True when the episode ended with status=failed."""
        return self.done and self.state.status == "failed"

    def progress_pct(self) -> str:
        """Return progress as a human-readable percentage string."""
        return f"{self.state.progress * 100:.1f}%"


# ---------------------------------------------------------------------------
# StepResult model — full return value of step()
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    """
    Complete return value of environment.step(action).

    Fields:
        observation:  Updated observation after the action was applied.
        reward:       Per-step reward value in [-1.0, 1.0].
        done:         True if the episode has ended.
        info:         Dict with extra context:
                        - "result":      action result code
                        - "error":       error detail (if any)
                        - "detail":      success detail message
                        - "milestone":   milestone name (if reached)
                        - "blockers":    remaining blockers (if finalize failed)
                        - "final_score": grader score [0.0, 1.0] (if done=True)
                        - "status":      episode status (if done=True)
    """

    observation: Observation        = Field(..., description="Updated observation post-action")
    reward:      float              = Field(..., description="Per-step reward in [-1.0, 1.0]")
    done:        bool               = Field(..., description="Episode termination flag")
    info:        dict[str, Any]     = Field(default_factory=dict, description="Extra context dict")

    @field_validator("reward")
    @classmethod
    def clamp_step_reward(cls, v: float) -> float:
        return round(max(-1.0, min(1.0, v)), 4)

    def final_score(self) -> float | None:
        """Return grader final score if episode is done, else None."""
        return self.info.get("final_score")

    def had_error(self) -> bool:
        """True if the action produced an error."""
        return "error" in self.info


# ---------------------------------------------------------------------------
# ResetResult model — return value of reset()
# ---------------------------------------------------------------------------


class ResetResult(BaseModel):
    """
    Return value of environment.reset(task_name).

    Fields:
        observation:  Initial observation for the new episode.
        task_name:    Which task was loaded.
        session_id:   Unique session identifier for this episode.
                      Must be passed back in all subsequent step() calls
                      when using the HTTP API (main.py).
        task_info:    Metadata about the loaded task.
    """

    observation: Observation = Field(..., description="Initial episode observation")
    task_name:   TaskName    = Field(..., description="Task that was loaded")
    session_id:  str         = Field(..., description="Unique session ID for this episode")
    task_info:   "TaskInfo"  = Field(..., description="Metadata about the loaded task")


# ---------------------------------------------------------------------------
# TaskInfo model — describes a single task
# ---------------------------------------------------------------------------


class TaskInfo(BaseModel):
    """
    Metadata about a single environment task.
    Embedded in ResetResult and used in openenv.yaml task registry.

    Fields:
        name:                  Task identifier.
        description:           Human-readable description.
        countries:             Destination countries for this task.
        max_steps:             Step limit before auto-fail.
        expected_score_range:  Tuple[min, max] — expected baseline agent score.
        departments_required:  Which departments must approve for full score.
        key_rules:             List of important rules the agent must know.
    """

    name:                 TaskName        = Field(..., description="Task identifier")
    description:          str             = Field(..., description="Human-readable task description")
    countries:            list[str]       = Field(..., description="Destination countries")
    max_steps:            int             = Field(..., gt=0, description="Max steps before auto-fail")
    expected_score_range: tuple[float, float] = Field(..., description="(min, max) expected baseline score")
    departments_required: list[str]       = Field(default_factory=list, description="Departments that must approve")
    key_rules:            list[str]       = Field(default_factory=list, description="Important rules for this task")

    @field_validator("expected_score_range")
    @classmethod
    def score_range_valid(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"expected_score_range must satisfy 0.0 <= lo <= hi <= 1.0, got {v}"
            )
        return v


# ---------------------------------------------------------------------------
# EpisodeSummary model — end-of-episode report
# ---------------------------------------------------------------------------


class EpisodeSummary(BaseModel):
    """
    Summary produced at the end of an episode (when done=True).
    Returned in the info dict of the final StepResult and used by graders.

    Fields:
        task_name:          Which task was run.
        session_id:         Session identifier.
        status:             "success" or "failed".
        final_score:        Grader score in [0.0, 1.0].
        steps_taken:        Total steps used.
        cumulative_reward:  Sum of all per-step rewards, clamped to [0.0, 1.0].
        verified_documents: Documents that were verified during the episode.
        departments_approved: Departments that approved.
        compliance_completed: Compliance items that were completed.
        remaining_blockers: Any blockers still unresolved (empty on success).
        action_history:     Full list of "action_type:target" strings taken.
    """

    task_name:             TaskName    = Field(..., description="Task that was run")
    session_id:            str         = Field(..., description="Session identifier")
    status:                EpisodeStatus = Field(..., description="Episode outcome")
    final_score:           float       = Field(..., ge=0.0, le=1.0, description="Grader score")
    steps_taken:           int         = Field(..., ge=0, description="Total steps used")
    cumulative_reward:     float       = Field(..., ge=0.0, le=1.0, description="Accumulated reward")
    verified_documents:    list[str]   = Field(default_factory=list)
    departments_approved:  list[str]   = Field(default_factory=list)
    compliance_completed:  list[str]   = Field(default_factory=list)
    remaining_blockers:    list[str]   = Field(default_factory=list)
    action_history:        list[str]   = Field(default_factory=list)

    @field_validator("final_score", "cumulative_reward")
    @classmethod
    def clamp_scores(cls, v: float) -> float:
        return round(max(0.0, min(1.0, v)), 4)

    def passed(self) -> bool:
        """True if the episode ended with success status."""
        return self.status == "success"

    def efficiency_ratio(self) -> float:
        """
        Score per step — higher means the agent solved the task more efficiently.
        Returns 0.0 if no steps were taken.
        """
        if self.steps_taken == 0:
            return 0.0
        return round(self.final_score / self.steps_taken, 4)


# ---------------------------------------------------------------------------
# HTTP request/response models for main.py (FastAPI endpoints)
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    """Request body for POST /reset."""
    task_name:  TaskName = Field(default="easy", description="Task to load")
    session_id: str | None = Field(default=None, description="Optional session ID — generated if not provided")


class StepRequest(BaseModel):
    """Request body for POST /step."""
    session_id: str    = Field(..., description="Session ID from reset()")
    action:     Action = Field(..., description="Action to execute")


class StateRequest(BaseModel):
    """Query parameters for GET /state."""
    session_id: str = Field(..., description="Session ID from reset()")


class HealthResponse(BaseModel):
    """Response for GET / — used by HuggingFace Space automated ping."""
    status: Literal["ok"]  = Field(default="ok")
    name:   str            = Field(default="openenv-workforce")
    version: str           = Field(default="1.0.0")


# ---------------------------------------------------------------------------
# Update forward references (needed for ResetResult → TaskInfo)
# ---------------------------------------------------------------------------

ResetResult.model_rebuild()
