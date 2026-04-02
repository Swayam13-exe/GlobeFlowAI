import json
from typing import Dict, Any, Tuple
from pydantic import ValidationError

from .models import EnvironmentState, Action
from .tasks import load_task, TASKS
from .validators import simulate_verify_document
from .rules import get_country_rules
from .reward import (
    REWARD_CORRECT_ACTION, 
    REWARD_PROGRESS, 
    REWARD_MILESTONE, 
    REWARD_CONFLICT_RESOLUTION,
    REWARD_FINAL_SUCCESS,
    PENALTY_WRONG_ACTION,
    PENALTY_RULE_VIOLATION,
    PENALTY_REPEATED_ACTION
)

class GlobeFlowEnv:
    def __init__(self, task_level: str = "easy"):
        self.task_level = task_level
        self._state: EnvironmentState = None
        self.reset()
        
    def reset(self, task_level: str = None) -> Dict[str, Any]:
        if task_level is not None:
            self.task_level = task_level
        self._state = load_task(self.task_level)
        return self._state.model_dump()
        
    def state(self) -> Dict[str, Any]:
        return self._state.model_dump()
        
    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        # validate action
        try:
            action = Action(**action_dict)
        except ValidationError as e:
            # Invalid action format -> penalty
            self._state.progress -= 0.1
            return self._state.model_dump(), PENALTY_WRONG_ACTION, False, {"error": "Invalid action format"}
            
        reward = 0.0
        done = False
        info = {}
        
        # Check repeated
        action_str = f"{action.action_type}_{action.target}"
        if action_str in self._state.previous_actions:
            reward += PENALTY_REPEATED_ACTION
            return self._state.model_dump(), reward, done, {"msg": "Repeated action penalty"}
            
        self._state.previous_actions.append(action_str)
        
        # Action logic
        act_type = action.action_type
        target = action.target
        
        reward += REWARD_CORRECT_ACTION 
        rules = get_country_rules(self._state.countries)
        
        if act_type == "request_document":
            if target in self._state.documents:
                doc = self._state.documents[target]
                if doc.status == "missing":
                    doc.status = "submitted"
                    reward += REWARD_PROGRESS
                else:
                    reward += PENALTY_WRONG_ACTION
            else:
                reward += PENALTY_WRONG_ACTION
                
        elif act_type == "verify_document":
            if simulate_verify_document(self._state, target):
                reward += REWARD_PROGRESS
            else:
                reward += PENALTY_WRONG_ACTION
                
        elif act_type == "approve_hr":
            self._state.departments.HR = True
            reward += REWARD_MILESTONE
            
        elif act_type == "approve_legal":
            if self._state.departments.HR and all(d.status == "verified" for d in self._state.documents.values()):
                self._state.departments.Legal = True
                reward += REWARD_MILESTONE
            else:
                reward += PENALTY_RULE_VIOLATION
                info["error"] = "Legal requires HR approval and all docs verified."
                
        elif act_type == "approve_finance":
            if self._state.departments.Legal:
                self._state.departments.Finance = True
                reward += REWARD_MILESTONE
            else:
                reward += PENALTY_RULE_VIOLATION
                info["error"] = "Finance requires Legal approval."
                
        elif act_type == "set_payroll":
            self._state.compliance.payroll = True
            reward += REWARD_PROGRESS
            
        elif act_type == "set_tax_id":
            self._state.compliance.tax_id = True
            reward += REWARD_PROGRESS
            # Check violation
            if rules.get("tax_forbidden"):
                reward += PENALTY_RULE_VIOLATION
                info["error"] = "Tax is forbidden in one of the countries (UAE)."
                
        elif act_type == "set_shadow_payroll":
            self._state.compliance.shadow_payroll = True
            reward += REWARD_PROGRESS
            
        elif act_type == "set_pdpa":
            self._state.compliance.pdpa = True
            reward += REWARD_PROGRESS
            
        elif act_type == "finalize_case":
            done = True
            if target == "rejected":
                self._state.status = "failed"
                # Check conflict scenario
                if "UAE" in self._state.countries and "Germany" in self._state.countries:
                    reward += REWARD_CONFLICT_RESOLUTION
            else:
                self._state.status = "success"
                
            grader_func = TASKS[self.task_level].grader
            final_score = grader_func(self._state)
            
            if final_score == 1.0:
                reward += REWARD_FINAL_SUCCESS
            info["score"] = final_score
            
        else:
            reward += PENALTY_WRONG_ACTION
            
        return self._state.model_dump(), reward, done, info
