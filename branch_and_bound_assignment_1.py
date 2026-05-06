"""
Backward-compatible wrapper for old scripts.

Old imports like
    import branch_and_bound_assignment_1 as bnb_module
continue to work, while the actual implementation lives in
    methods.branch_and_bound.core
"""

from methods.branch_and_bound.core import *  # noqa: F401,F403
