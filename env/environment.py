"""
env/environment.py
==================
Core OpenEnv environment class for openenv-workforce.

Implements the full OpenEnv interface:
  - reset(task_name)  → Observation
  - step(action)      → StepResult
  - state()           → WorkforceState

FIX (v1.1):
  _handle_finalize_case no longer sets status = "success" directly.
  It now calls the grader, compares the score against the task's
  expected score range, and sets status = "success" | "failed"
  based on that comparison. The grader score drives the outcome.

All state is held in-memory on the instance (self._state).
No external database, no network calls, fully deterministic.

Author: Team AI Kalesh
"""

from __future__ import annotations

import copy
import uuid
from typing import Any

from env.models import (
    Action,
    ComplianceStatus,
    DepartmentStatus,
    DocumentRecord,
    EmployeeRecord,
    Observation,
    Reward,
    StepResult,
    WorkforceState,
)
from env.reward import REWARDS, compute_reward
from env.rules import (
    COUNTRY_RULES,
    DEPARTMENT_DEPENDENCIES,
    REQUIRED_DOCUMENTS,
)
from env.validators import (
    validate_compliance_action,
    validate_department_prerequisites,
    validate_document,
)
from env.tasks import TASK_INFO, TASKS


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
    "finalize_case",
}

VALID_DOCUMENTS: set[str] = {
    "passport",
    "visa",
    "employment_letter",
    "degree_certificate",
}
VALID_DEPARTMENTS: set[str] = {"HR", "Legal", "Finance"}


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------


class WorkforceEnv:
    """
    Stateful OpenEnv environment simulating employee relocation workflows.

    An AI agent interacts with this environment via:
        reset(task_name)  → initial Observation
        step(action)      → StepResult (observation, reward, done, info)
        state()           → current WorkforceState

    Episode termination conditions:
        finalize_case + score in range  → status="success", done=True
        finalize_case + score out range → status="failed",  done=True
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

        Raises:
            ValueError: If task_name is not recognised.
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
        if self._done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"error": "episode_already_done"},
            )

        prev_progress = self._state["progress"]

        # ── Validate action format ──────────────────────────────────────────
        format_error = self._validate_action_format(action)
        if format_error:
            reward_val = REWARDS["invalid_action"]
            self._last_action_result = "invalid_action"
            self._last_action_error = format_error
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
                info={"error": format_error},
            )

        # ── Check for repeated action ───────────────────────────────────────
        action_key = f"{action.action_type}:{action.target}"
        if action_key in self._state["previous_actions"]:
            reward_val = REWARDS["repeated_action"]
            self._last_action_result = "repeated_action"
            self._last_action_error = f"Action '{action_key}' already performed."
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
                info={"warning": "repeated_action"},
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

        # ── Check deadline/step-limit termination ───────────────────────────
        # finalize_case sets self._done directly inside _handle_finalize_case.
        # _check_terminal_conditions only handles deadline + step-limit paths.
        if not self._done:
            self._check_terminal_conditions()

        # ── Build info payload ──────────────────────────────────────────────
        info: dict[str, Any] = {"result": result}
        if error:
            info["error"] = error
        if extra_info:
            info.update(extra_info)

        # Attach grader score and final status whenever the episode ends
        if self._done and "final_score" not in info:
            from env.graders import grade
            info["final_score"] = grade(self._task_name, self._state)
            info["status"]      = self._state["status"]

        return StepResult(
            observation=self._build_observation(),
            reward=max(-1.0, min(1.0, reward_val)),
            done=self._done,
            info=info,
        )

    def state(self) -> WorkforceState:
        """Return the current full environment state as a typed Pydantic model."""
        return self._dict_to_model()

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
            "set_payroll":        self._handle_set_payroll,
            "set_tax_id":         self._handle_set_tax_id,
            "set_shadow_payroll": self._handle_set_shadow_payroll,
            "set_pdpa":           self._handle_set_pdpa,
            "finalize_case":      self._handle_finalize_case,
        }
        return handlers[action.action_type](action)

    # -----------------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------------

    def _handle_request_document(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        doc_name = action.target
        if doc_name not in VALID_DOCUMENTS:
            return "invalid_action", f"Unknown document: {doc_name}", {}

        doc = self._state["documents"].get(doc_name)
        if not doc:
            return "invalid_action", f"Document '{doc_name}' not in this case", {}

        if doc["status"] != "missing":
            return "wrong_action", f"Document '{doc_name}' already {doc['status']}", {}

        self._state["documents"][doc_name]["status"] = "submitted"

        if doc_name == "visa":
            countries  = self._state["countries"]
            needs_visa = any(
                COUNTRY_RULES.get(c, {}).get("requires_visa", False)
                for c in countries
            )
            self._state["documents"]["visa"]["is_valid"] = needs_visa

        return "success", None, {"detail": f"Document '{doc_name}' submitted successfully"}

    def _handle_verify_document(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        doc_name = action.target
        if doc_name not in VALID_DOCUMENTS:
            return "invalid_action", f"Unknown document: {doc_name}", {}

        doc = self._state["documents"].get(doc_name)
        if not doc:
            return "invalid_action", f"Document '{doc_name}' not in this case", {}

        is_approved, reason = validate_document(self._state, doc_name)

        if is_approved:
            self._state["documents"][doc_name]["status"] = "verified"
            return "success", None, {"detail": f"Document '{doc_name}' verified successfully"}
        else:
            self._state["documents"][doc_name]["status"] = "rejected"
            return "wrong_action", f"Document '{doc_name}' rejected: {reason}", {"detail": reason}

    def _handle_approve_department(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        dept_map = {
            "approve_hr":      "HR",
            "approve_legal":   "Legal",
            "approve_finance": "Finance",
        }
        dept = dept_map.get(action.action_type)
        if not dept:
            return "invalid_action", f"Unknown department action: {action.action_type}", {}

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

    def _handle_set_payroll(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        country = action.target
        if country not in self._state["countries"]:
            return "invalid_action", f"Country '{country}' is not part of this relocation case", {}

        if self._state["compliance"]["payroll"]:
            return "wrong_action", "Payroll already configured", {}

        if not COUNTRY_RULES.get(country, {}).get("requires_payroll", False):
            return "rule_violation", f"Payroll not required for {country}", {}

        self._state["compliance"]["payroll"] = True
        return "success", None, {"detail": f"Payroll configured for {country}"}

    def _handle_set_tax_id(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        country = action.target

        ok, reason = validate_compliance_action(self._state, "set_tax_id", country)
        if not ok:
            return "rule_violation", reason, {}

        if country not in self._state["countries"]:
            return "invalid_action", f"Country '{country}' is not part of this relocation case", {}

        if self._state["compliance"]["tax_id"]:
            return "wrong_action", "Tax ID already registered", {}

        self._state["compliance"]["tax_id"] = True
        return "success", None, {"detail": f"Tax ID registered for {country}"}

    def _handle_set_shadow_payroll(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        country = action.target
        ok, reason = validate_compliance_action(self._state, "set_shadow_payroll", country)
        if not ok:
            return "rule_violation", reason, {}

        if country not in self._state["countries"]:
            return "invalid_action", f"Country '{country}' is not part of this relocation case", {}

        if self._state["compliance"]["shadow_payroll"]:
            return "wrong_action", "Shadow payroll already enabled", {}

        self._state["compliance"]["shadow_payroll"] = True
        return "success", None, {"detail": "Shadow payroll enabled for Singapore"}

    def _handle_set_pdpa(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        country = action.target
        ok, reason = validate_compliance_action(self._state, "set_pdpa", country)
        if not ok:
            return "rule_violation", reason, {}

        if country not in self._state["countries"]:
            return "invalid_action", f"Country '{country}' is not part of this relocation case", {}

        if self._state["compliance"]["pdpa"]:
            return "wrong_action", "PDPA consent already collected", {}

        self._state["compliance"]["pdpa"] = True
        return "success", None, {"detail": "PDPA consent collected for Singapore"}

    # -----------------------------------------------------------------------
    # _handle_finalize_case — FIXED METHOD (v1.1)
    # -----------------------------------------------------------------------

    def _handle_finalize_case(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        """
        Attempt to close the relocation case and evaluate final performance.

        Evaluation Flow:
          1. Check for blockers (pure rule enforcement).
          2. Compute final score using the deterministic grader.
          3. Retrieve the expected score range from TASK_INFO.
          4. Compare score vs range to determine success/failure.
          5. Terminate the episode.
        """
        if action.target != "all":
            return (
                "invalid_action",
                f"finalize_case target must be 'all', got '{action.target}'",
                {},
            )

        # ── Step 1: Blocker check ───────────────────────────────────────────
        blockers = self._get_blockers()
        if blockers:
            return (
                "rule_violation",
                "Cannot finalize — mandatory blockers remain",
                {"blockers": blockers},
            )

        # ── Step 2: Run grader to get numeric score ─────────────────────────
        from env.graders import grade
        final_score: float = grade(self._task_name, self._state)

        # ── Step 3: Compare score against expected range from TASK_INFO ─────
        task_meta = TASK_INFO.get(self._task_name, {})
        min_score, max_score = task_meta.get("expected_score_range", (0.0, 1.0))
        
        score_in_range: bool = min_score <= final_score <= max_score

        # ── Step 4: Set status based on score comparison ────────────────────
        self._state["status"] = "success" if score_in_range else "failed"

        # ── Step 5: End the episode ─────────────────────────────────────────
        self._done = True

        # ── Build rich info payload ─────────────────────────────────────────
        extra: dict[str, Any] = {
            "final_score":    final_score,
            "expected_range": [min_score, max_score],
            "score_in_range": score_in_range,
            "status":         self._state["status"],
            "milestone":      "episode_complete",
        }

        if score_in_range:
            extra["detail"] = (
                f"Evaluation: Success. Score {final_score:.4f} is within range."
            )
            return "success", None, extra
        else:
            if final_score < min_score:
                reason = f"Evaluation: Failed. Under-performance (Score {final_score:.4f} < {min_score})."
            else:
                reason = f"Evaluation: Failed. Over-optimized (Score {final_score:.4f} > {max_score})."
            
            extra["detail"] = reason
            return "rule_violation", reason, extra

    # -----------------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------------

    def _validate_action_format(self, action: Action) -> str | None:
        """Return an error string if action is structurally invalid, else None."""
        if action.action_type not in VALID_ACTION_TYPES:
            return (
                f"Unknown action_type '{action.action_type}'. "
                f"Valid types: {sorted(VALID_ACTION_TYPES)}"
            )
        if not action.target or not isinstance(action.target, str):
            return "Target must be a non-empty string"
        if action.action_type in {"request_document", "verify_document"}:
            if action.target not in VALID_DOCUMENTS:
                return (
                    f"Unknown document '{action.target}'. "
                    f"Valid documents: {sorted(VALID_DOCUMENTS)}"
                )
        if action.action_type in {"approve_hr", "approve_legal", "approve_finance"}:
            if action.target not in VALID_DEPARTMENTS:
                return (
                    f"Unknown department '{action.target}'. "
                    f"Valid departments: {sorted(VALID_DEPARTMENTS)}"
                )
        return None

    # -----------------------------------------------------------------------
    # Progress + blocker computation
    # -----------------------------------------------------------------------

    def _compute_progress(self) -> float:
        """Compute progress — delegates to rules.compute_checklist."""
        from env.rules import compute_checklist
        return compute_checklist(self._state)["progress"]

    def _get_blockers(self) -> list[str]:
        """Return all blockers — delegates to rules.get_blockers."""
        from env.rules import get_blockers
        return get_blockers(self._state)

    # -----------------------------------------------------------------------
    # Terminal condition check — deadline and step limit only
    # -----------------------------------------------------------------------

    def _check_terminal_conditions(self) -> None:
        """
        Handle deadline and step-limit termination.
        finalize_case termination is handled inside _handle_finalize_case.
        """
        if self._done:
            return

        if self._state["deadline_days"] <= 0:
            self._state["status"] = "failed"
            self._done = True
            self._last_action_error = "Episode failed: deadline reached"

        elif self._steps_taken >= MAX_STEPS:
            self._state["status"] = "failed"
            self._done = True
            self._last_action_error = (
                f"Episode failed: max steps ({MAX_STEPS}) reached"
            )

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
        """Return action strings the agent can usefully take. Format: 'action_type:target'"""
        available: list[str] = []
        countries = self._state["countries"]
        docs      = self._state["documents"]
        depts     = self._state["departments"]
        comp      = self._state["compliance"]

        for doc_name, doc in docs.items():
            if doc["status"] == "missing":
                available.append(f"request_document:{doc_name}")
            elif doc["status"] == "submitted":
                available.append(f"verify_document:{doc_name}")

        if not depts["HR"]:
            available.append("approve_hr:HR")

        if not depts["Legal"]:
            ok, _ = validate_department_prerequisites(self._state, "Legal")
            if ok:
                available.append("approve_legal:Legal")

        if not depts["Finance"]:
            ok, _ = validate_department_prerequisites(self._state, "Finance")
            if ok:
                available.append("approve_finance:Finance")

        for country in countries:
            rules = COUNTRY_RULES.get(country, {})
            if rules.get("requires_payroll") and not comp["payroll"]:
                available.append(f"set_payroll:{country}")
            if rules.get("requires_tax_id") and not comp["tax_id"]:
                available.append(f"set_tax_id:{country}")
            if rules.get("requires_pdpa") and not comp["pdpa"]:
                available.append(f"set_pdpa:{country}")
            if rules.get("requires_shadow_payroll") and not comp["shadow_payroll"]:
                available.append(f"set_shadow_payroll:{country}")

        if not self._get_blockers():
            available.append("finalize_case:all")

        return available

    # -----------------------------------------------------------------------
    # State conversion
    # -----------------------------------------------------------------------

    def _dict_to_model(self) -> WorkforceState:
        """Convert internal state dict to a typed WorkforceState Pydantic model."""
        s = self._state
        return WorkforceState(
            case_id=s["case_id"],
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
            deadline_days=s["deadline_days"],
            previous_actions=s["previous_actions"],
            progress=s["progress"],
            status=s["status"],
        )