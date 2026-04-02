"""
env/graders.py
==============
Re-export shim for the graders module.

All actual grading logic lives in graders/graders.py.
This module exposes the canonical import path used by environment.py:

    from env.graders import grade

Author: Team AI Kalesh
"""

# Add graders/ to path so the import resolves correctly regardless of CWD

from graders.graders import (
    explain,
    grade,
    grade_all,
    print_score_table,
    GradeReport,
    grade_easy,
    grade_medium,
    grade_hard,
)