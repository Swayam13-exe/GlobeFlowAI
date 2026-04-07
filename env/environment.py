"""
env/environment.py
==================
Core OpenEnv environment class for openenv-workforce.

Implements the full OpenEnv interface:
  - reset(task_name)  → Observation
  - step(action)      → StepResult
  - state()           → WorkforceState

Author: Team AI Kalesh
"""

from __future__ import annotations

import copy
import uuid
from typing import Any

from env.models import (
    Action,
    ComplianceStatus,
    ConflictRecord,
    DepartmentStatus,
    DocumentRecord,
    EmployeeRecord,
    Observation,
    StepResult,
    WorkforceState,
)
from env.reward import REWARDS, compute_reward
from env.rules import COUNTRY_RULES, DEPARTMENT_DEPENDENCIES
from env.validators import (
    validate_compliance_action,
    validate_department_prerequisites,
    validate_document,
)
from env.tasks import TASKS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS: int = 25

VALID_ACTION_TYPES: set[str] = {
    "request_document",
    "verify_document",
    "approve_hr",
    "approve_legal",
    "approve_finance",
    "set_payroll",
    "set_tax_id",
    "set_shadow_payroll",
    "set_pdpa",
    "resolve_conflict",     # FIX 4: added for hard task
    "finalize_case",
}

# FIX 3: All documents used across ALL three tasks
VALID_DOCUMENTS: set[str] = {
    "passport",
    "visa",
    "employment_letter",
    "degree_certificate",
    "work_permit",          # Germany
    "employment_pass",      # Singapore
    "residence_permit",     # UAE
    "tax_form",
}

VALID_DEPARTMENTS: set[str] = {"HR", "Legal", "Finance"}

# Actions that legitimately have no target (empty string is fine)
NO_TARGET_ACTIONS: set[str] = {
    "approve_hr",
    "approve_legal",
    "approve_finance",
    "set_payroll",
    "set_tax_id",
    "set_shadow_payroll",
    "set_pdpa",
    "resolve_conflict",
    "finalize_case",
}


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------


class WorkforceEnv:
    """
    Stateful OpenEnv environment simulating employee relocation workflows.

    Episode termination conditions:
        finalize_case with no blockers  → status="success", done=True
        deadline_days reaches 0         → status="failed",  done=True
        steps_taken reaches MAX_STEPS   → status="failed",  done=True
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}
        self._task_name: str = ""
        self._steps_taken: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._last_action_result: str = "Episode not started. Call reset() first."
        self._last_action_error: str | None = None
        self._session_id: str = str(uuid.uuid4())

    # -----------------------------------------------------------------------
    # Public OpenEnv interface
    # -----------------------------------------------------------------------

    def reset(self, task_name: str = "easy") -> Observation:
        """
        Start a new episode for the given task.

        Args:
            task_name: One of "easy", "medium", "hard".

        Returns:
            Observation with the initial state and available actions.
        """
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Valid tasks: {list(TASKS.keys())}"
            )

        self._state = copy.deepcopy(TASKS[task_name])
        self._task_name = task_name
        self._steps_taken = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._last_action_result = "Episode started. Make your first action."
        self._last_action_error = None
        self._session_id = str(uuid.uuid4())

        # Ensure conflicts key exists (empty list for easy/medium)
        if "conflicts" not in self._state:
            self._state["conflicts"] = []

        # Ensure required lists exist
        if "required_departments" not in self._state:
            self._state["required_departments"] = []
        if "required_compliance" not in self._state:
            self._state["required_compliance"] = []

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Apply one agent action and return the resulting state + reward.

        Args:
            action: Action(action_type=str, target=str)

        Returns:
            StepResult with updated observation, per-step reward, done flag,
            and an info dict containing the grader score if done=True.
        """
        # Guard: episode already finished
        if self._done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"error": "episode_already_done"},
            )

        prev_progress = self._state.get("progress", 0.0)

        # ── Validate action format ──────────────────────────────────────────
        format_error = self._validate_action_format(action)
        if format_error:
            return self._penalise(
                result="invalid_action",
                error=format_error,
                reward_key="invalid_action",
            )

        # ── Check for repeated action ───────────────────────────────────────
        action_key = action.to_key()
        if action_key in self._state["previous_actions"]:
            return self._penalise(
                result="repeated_action",
                error=f"Action '{action_key}' already performed.",
                reward_key="repeated_action",
            )

        # ── Dispatch to action handler ──────────────────────────────────────
        result, error, extra_info = self._dispatch(action)

        # ── Compute per-step reward ─────────────────────────────────────────
        reward_val = compute_reward(
            action=action,
            result=result,
            state=self._state,
            prev_progress=prev_progress,
        )

        # ── Record action in history ────────────────────────────────────────
        self._state["previous_actions"].append(action_key)

        # ── Update progress ─────────────────────────────────────────────────
        self._state["progress"] = self._compute_progress()

        # ── Tick clock ──────────────────────────────────────────────────────
        self._steps_taken += 1
        self._state["deadline_days"] = max(0, self._state["deadline_days"] - 1)

        # ── Accumulate episode reward ───────────────────────────────────────
        self._cumulative_reward = max(
            0.0, min(1.0, self._cumulative_reward + reward_val)
        )

        # ── Update last action metadata ─────────────────────────────────────
        self._last_action_result = result
        self._last_action_error = error

        # ── Check deadline / step-limit termination ─────────────────────────
        # finalize_case sets self._done inside _handle_finalize_case.
        if not self._done:
            self._check_terminal_conditions()

        # ── Build info payload ──────────────────────────────────────────────
        info: dict[str, Any] = {"result": result}
        if error:
            info["error"] = error
        if extra_info:
            info.update(extra_info)

        # Always attach grader score when episode ends
        if self._done and "final_score" not in info:
            from env.graders import grade
            info["final_score"] = grade(self._task_name, self._state)
            info["status"] = self._state["status"]

        return StepResult(
            observation=self._build_observation(),
            reward=max(-1.0, min(1.0, reward_val)),
            done=self._done,
            info=info,
        )

    def state(self) -> WorkforceState:
        """Return the current full environment state as a typed Pydantic model."""
        ws = self._dict_to_model()
        # Attach episode metadata into previous_actions as a convention
        # (WorkforceState has no step/reward fields — those live in environment)
        return ws

    # -----------------------------------------------------------------------
    # Penalty helper (for invalid / repeated actions)
    # -----------------------------------------------------------------------

    def _penalise(
        self, result: str, error: str, reward_key: str
    ) -> StepResult:
        """
        Apply a penalty step without mutating case state.
        Still ticks the clock and accumulates reward.
        """
        reward_val = REWARDS.get(reward_key, -0.2)
        self._last_action_result = result
        self._last_action_error = error
        self._steps_taken += 1
        self._state["deadline_days"] = max(0, self._state["deadline_days"] - 1)
        self._cumulative_reward = max(
            0.0, min(1.0, self._cumulative_reward + reward_val)
        )
        self._check_terminal_conditions()
        return StepResult(
            observation=self._build_observation(),
            reward=max(-1.0, min(1.0, reward_val)),
            done=self._done,
            info={"result": result, "error": error},
        )

    # -----------------------------------------------------------------------
    # Action dispatcher
    # -----------------------------------------------------------------------

    def _dispatch(
        self, action: Action
    ) -> tuple[str, str | None, dict[str, Any]]:
        """Route action to the correct handler."""
        handlers = {
            "request_document":   self._handle_request_document,
            "verify_document":    self._handle_verify_document,
            "approve_hr":         self._handle_approve_department,
            "approve_legal":      self._handle_approve_department,
            "approve_finance":    self._handle_approve_department,
            "set_payroll":        self._handle_set_compliance,
            "set_tax_id":         self._handle_set_compliance,
            "set_shadow_payroll": self._handle_set_compliance,
            "set_pdpa":           self._handle_set_compliance,
            "resolve_conflict":   self._handle_resolve_conflict,   # FIX 4
            "finalize_case":      self._handle_finalize_case,
        }
        handler = handlers.get(action.action_type)
        if not handler:
            return "invalid_action", f"No handler for '{action.action_type}'", {}
        return handler(action)

    # -----------------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------------

    def _handle_request_document(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        doc_name = action.target
        doc = self._state["documents"].get(doc_name)
        if not doc:
            return "invalid_action", f"Document '{doc_name}' not in this case", {}
        if doc["status"] != "missing":
            return "wrong_action", f"Document '{doc_name}' already {doc['status']}", {}

        self._state["documents"][doc_name]["status"] = "submitted"
        self._state["documents"][doc_name]["is_valid"] = True  # deterministic
        return "success", None, {"detail": f"Document '{doc_name}' submitted"}

    def _handle_verify_document(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        doc_name = action.target
        doc = self._state["documents"].get(doc_name)
        if not doc:
            return "invalid_action", f"Document '{doc_name}' not in this case", {}

        # Must be submitted before it can be verified
        status = doc.get("status", "missing")
        if status == "missing":
            return (
                "prereq_violated",
                f"Document '{doc_name}' has not been submitted yet. "
                f"Call request_document first.",
                {},
            )
        if status == "verified":
            return "wrong_action", f"Document '{doc_name}' is already verified", {}
        if status == "rejected":
            return (
                "wrong_action",
                f"Document '{doc_name}' was rejected. Re-request before verifying.",
                {},
            )

        is_approved, reason = validate_document(self._state, doc_name)

        if is_approved:
            self._state["documents"][doc_name]["status"] = "verified"
            return "success", None, {"detail": f"Document '{doc_name}' verified"}
        else:
            self._state["documents"][doc_name]["status"] = "rejected"
            return "wrong_action", f"Document '{doc_name}' rejected: {reason}", {}

    def _handle_approve_department(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        dept_map = {
            "approve_hr":      "HR",
            "approve_legal":   "Legal",
            "approve_finance": "Finance",
        }
        dept = dept_map[action.action_type]

        if self._state["departments"][dept]:
            return "wrong_action", f"{dept} already approved", {}

        ok, reason = validate_department_prerequisites(self._state, dept)
        if not ok:
            return "prereq_violated", reason, {}

        self._state["departments"][dept] = True
        extra: dict[str, Any] = {"detail": f"{dept} approved"}
        if all(self._state["departments"].values()):
            extra["milestone"] = "all_departments_approved"
        return "success", None, extra

    def _handle_set_compliance(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        """
        Unified handler for set_payroll, set_tax_id, set_shadow_payroll, set_pdpa.
        FIX 6: accepts empty target — infers country from case if needed.
        """
        action_type = action.action_type

        # Map action → compliance field
        compliance_field_map = {
            "set_payroll":        "payroll",
            "set_tax_id":         "tax_id",
            "set_shadow_payroll": "shadow_payroll",
            "set_pdpa":           "pdpa",
        }
        field = compliance_field_map[action_type]

        # Already set?
        if self._state["compliance"][field]:
            return "wrong_action", f"{field} already configured", {}

        # Validate using rules
        # FIX 6: use first relevant country if target is empty
        country = action.target or (
            self._state["countries"][0] if self._state["countries"] else ""
        )
        ok, reason = validate_compliance_action(self._state, action_type, country)
        if not ok:
            return "rule_violation", reason, {}

        self._state["compliance"][field] = True
        return "success", None, {"detail": f"{field} configured for {country}"}

    def _handle_resolve_conflict(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        """
        FIX 4: Handle resolve_conflict for hard task.
        Resolves the first unresolved conflict in the list.
        """
        conflicts = self._state.get("conflicts", [])
        unresolved = [c for c in conflicts if not c.get("resolved", False)]

        if not unresolved:
            return "wrong_action", "No unresolved conflicts in this case", {}

        # Resolve the first unresolved conflict
        for c in self._state["conflicts"]:
            if not c.get("resolved", False):
                c["resolved"] = True
                return "success", None, {
                    "detail": f"Conflict resolved: {c['rule']}",
                    "milestone": "conflict_resolved",
                }

        return "wrong_action", "No conflicts to resolve", {}

    def _handle_finalize_case(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        """
        Attempt to close the relocation case.

        FIX 1: Accept both target="" and target="all".
        FIX 7: Always set status="success" when blockers pass —
                do NOT gate on expected_score_range (that's documentation only).
        """
        # FIX 1: accept empty target OR "all"
        if action.target not in ("", "all"):
            return (
                "invalid_action",
                f"finalize_case target must be '' or 'all', got '{action.target}'",
                {},
            )

        # Blocker check
        blockers = self._get_blockers()
        if blockers:
            return (
                "rule_violation",
                "Cannot finalize — mandatory blockers remain",
                {"blockers": blockers},
            )

        # Run grader for the score
        from env.graders import grade
        final_score: float = grade(self._task_name, self._state)

        # FIX 7: success is determined by passing blockers, not score range
        self._state["status"] = "success"
        self._done = True

        return "success", None, {
            "final_score": final_score,
            "status":      "success",
            "milestone":   "episode_complete",
            "detail":      f"Case finalized. Score: {final_score:.4f}",
        }

    # -----------------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------------

    def _validate_action_format(self, action: Action) -> str | None:
        """Return error string if action is structurally invalid, else None."""
        if action.action_type not in VALID_ACTION_TYPES:
            return (
                f"Unknown action_type '{action.action_type}'. "
                f"Valid: {sorted(VALID_ACTION_TYPES)}"
            )

        # FIX 2: document actions need a non-empty target from VALID_DOCUMENTS
        if action.action_type in {"request_document", "verify_document"}:
            if not action.target:
                return "Document actions require a non-empty target"
            if action.target not in VALID_DOCUMENTS:
                return (
                    f"Unknown document '{action.target}'. "
                    f"Valid: {sorted(VALID_DOCUMENTS)}"
                )

        # FIX 2: all other actions allow empty target — no validation needed
        return None

    # -----------------------------------------------------------------------
    # Progress + blocker computation
    # -----------------------------------------------------------------------

    def _compute_progress(self) -> float:
        """Compute weighted progress fraction [0.0, 1.0]."""
        docs  = self._state.get("documents", {})
        depts = self._state.get("departments", {})
        comp  = self._state.get("compliance", {})
        conflicts = self._state.get("conflicts", [])
        req_depts = self._state.get("required_departments", list(depts.keys()))
        req_comp  = self._state.get("required_compliance", [])

        # Document progress: verified=1.0, submitted=0.5, else=0.0
        if docs:
            doc_scores = []
            for d in docs.values():
                if d["status"] == "verified":
                    doc_scores.append(1.0)
                elif d["status"] == "submitted":
                    doc_scores.append(0.5)
                else:
                    doc_scores.append(0.0)
            doc_prog = sum(doc_scores) / len(doc_scores)
        else:
            doc_prog = 0.0

        # Department progress
        if req_depts:
            dept_prog = sum(1 for d in req_depts if depts.get(d, False)) / len(req_depts)
        else:
            dept_prog = 1.0

        # Compliance progress
        if req_comp:
            comp_prog = sum(1 for c in req_comp if comp.get(c, False)) / len(req_comp)
        else:
            comp_prog = 1.0

        # Conflict resolution progress
        if conflicts:
            conflict_prog = sum(1 for c in conflicts if c.get("resolved", False)) / len(conflicts)
        else:
            conflict_prog = 1.0

        total = (
            0.40 * doc_prog
            + 0.35 * dept_prog
            + 0.15 * comp_prog
            + 0.10 * conflict_prog
        )
        return round(min(max(total, 0.0), 1.0), 4)

    def _get_blockers(self) -> list[str]:
        """Return human-readable list of reasons finalize_case cannot proceed."""
        blockers: list[str] = []
        docs      = self._state.get("documents", {})
        depts     = self._state.get("departments", {})
        comp      = self._state.get("compliance", {})
        conflicts = self._state.get("conflicts", [])
        req_depts = self._state.get("required_departments", [])
        req_comp  = self._state.get("required_compliance", [])

        # All documents must be verified
        unverified = [k for k, v in docs.items() if v["status"] != "verified"]
        if unverified:
            blockers.append(f"Unverified documents: {unverified}")

        # Required departments must approve
        missing_depts = [d for d in req_depts if not depts.get(d, False)]
        if missing_depts:
            blockers.append(f"Pending department approvals: {missing_depts}")

        # Required compliance must be set
        missing_comp = [c for c in req_comp if not comp.get(c, False)]
        if missing_comp:
            blockers.append(f"Incomplete compliance items: {missing_comp}")

        # All conflicts must be resolved
        unresolved = [c["rule"] for c in conflicts if not c.get("resolved", False)]
        if unresolved:
            blockers.append(f"Unresolved conflicts: {unresolved}")

        return blockers

    # -----------------------------------------------------------------------
    # Terminal condition check
    # -----------------------------------------------------------------------

    def _check_terminal_conditions(self) -> None:
        """Handle deadline and step-limit termination only."""
        if self._done:
            return
        if self._state.get("deadline_days", 1) <= 0:
            self._state["status"] = "failed"
            self._done = True
            self._last_action_error = "Episode failed: deadline reached"
        elif self._steps_taken >= MAX_STEPS:
            self._state["status"] = "failed"
            self._done = True
            self._last_action_error = f"Episode failed: max steps ({MAX_STEPS}) reached"

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Construct the Observation returned to the agent after every step."""
        return Observation(
            state=self._dict_to_model(),
            available_actions=self._get_available_actions(),
            current_blockers=self._get_blockers(),
            last_action_result=self._last_action_result,
            last_action_error=self._last_action_error,
            steps_taken=self._steps_taken,
            done=self._done,
        )

    def _get_available_actions(self) -> list[str]:
        """Return useful actions the agent can take. Format: 'action_type:target'"""
        available: list[str] = []
        docs      = self._state.get("documents", {})
        depts     = self._state.get("departments", {})
        comp      = self._state.get("compliance", {})
        countries = self._state.get("countries", [])
        conflicts = self._state.get("conflicts", [])

        # Document actions
        for doc_name, doc in docs.items():
            if doc["status"] == "missing":
                available.append(f"request_document:{doc_name}")
            elif doc["status"] == "submitted":
                available.append(f"verify_document:{doc_name}")

        # Department actions
        if not depts.get("HR", False):
            available.append("approve_hr")
        if not depts.get("Legal", False):
            ok, _ = validate_department_prerequisites(self._state, "Legal")
            if ok:
                available.append("approve_legal")
        if not depts.get("Finance", False):
            ok, _ = validate_department_prerequisites(self._state, "Finance")
            if ok:
                available.append("approve_finance")

        # Compliance actions
        for country in countries:
            rules = COUNTRY_RULES.get(country, {})
            if rules.get("requires_payroll") and not comp.get("payroll"):
                available.append(f"set_payroll:{country}")
            if rules.get("requires_tax_id") and not comp.get("tax_id"):
                available.append(f"set_tax_id:{country}")
            if rules.get("requires_pdpa") and not comp.get("pdpa"):
                available.append(f"set_pdpa:{country}")
            if rules.get("requires_shadow_payroll") and not comp.get("shadow_payroll"):
                available.append(f"set_shadow_payroll:{country}")

        # Conflict resolution
        unresolved = [c for c in conflicts if not c.get("resolved", False)]
        if unresolved:
            available.append("resolve_conflict")

        # Finalize
        if not self._get_blockers():
            available.append("finalize_case")

        return available

    # -----------------------------------------------------------------------
    # State conversion: dict → WorkforceState Pydantic model
    # -----------------------------------------------------------------------

    def _dict_to_model(self) -> WorkforceState:
        """Convert internal state dict to a typed WorkforceState Pydantic model."""
        s = self._state
        if not s:
            # Return minimal state if reset() hasn't been called yet
            return WorkforceState(
                case_id="none",
                task_name="easy",
                employee=EmployeeRecord(role="Engineer", has_dependents=False),
                countries=["Germany"],
            )

        return WorkforceState(
            case_id=s["case_id"],
            task_name=s.get("task_name", self._task_name),   # FIX 5
            employee=EmployeeRecord(
                role=s["employee"]["role"],
                has_dependents=s["employee"]["has_dependents"],
            ),
            countries=s["countries"],
            documents={
                name: DocumentRecord(
                    status=doc["status"],
                    is_valid=doc["is_valid"],
                )
                for name, doc in s["documents"].items()
            },
            departments=DepartmentStatus(
                HR=s["departments"]["HR"],
                Legal=s["departments"]["Legal"],
                Finance=s["departments"]["Finance"],
            ),
            compliance=ComplianceStatus(
                tax_id=s["compliance"]["tax_id"],
                payroll=s["compliance"]["payroll"],
                pdpa=s["compliance"]["pdpa"],
                shadow_payroll=s["compliance"]["shadow_payroll"],
            ),
            conflicts=[                                        # FIX 5
                ConflictRecord(
                    countries=c["countries"],
                    rule=c["rule"],
                    resolved=c.get("resolved", False),
                )
                for c in s.get("conflicts", [])
            ],
            deadline_days=s["deadline_days"],
            previous_actions=s["previous_actions"],
            progress=s.get("progress", 0.0),
            status=s["status"],
            required_departments=s.get("required_departments", []),  # FIX 5
            required_compliance=s.get("required_compliance", []),     # FIX 5
        )