"""Optional genetic-programming architecture-policy package.

Install ``requirements-gp.txt`` before importing the DEAP-backed submodules.
The lightweight configuration remains importable by non-GP workflows.
"""

from .config import GPArchitectureConfig

__all__ = ["GPArchitectureConfig"]
