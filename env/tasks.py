from .data import get_base_state
from .graders import grade_easy, grade_medium, grade_hard
from .models import EnvironmentState

class Task:
    def __init__(self, name: str, level: str, grader_func):
        self.name = name
        self.level = level
        self.grader = grader_func

TASKS = {
    "easy": Task("India -> Germany", "easy", grade_easy),
    "medium": Task("India -> Singapore", "medium", grade_medium),
    "hard": Task("India -> Germany + UAE", "hard", grade_hard)
}

def load_task(level: str) -> EnvironmentState:
    if level not in TASKS:
        raise ValueError(f"Unknown task level: {level}")
    return get_base_state(level)
