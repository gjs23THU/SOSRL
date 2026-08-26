"""Dynamic architecture-scheduling NSGA-II baseline."""

from .config import NSGA2Config
from .decoder import DynamicScheduleDecoder
from .model import Chromosome, DecodeResult, ProblemLayout
from .artifacts import solve_manifest_nsga2
from .solver import (
    NSGA2RunResult,
    NSGA2ScenarioResult,
    crowding_distance,
    run_nsga2,
    select_representatives,
    solve_scenario_nsga2,
)

__all__ = [
    "Chromosome",
    "DecodeResult",
    "DynamicScheduleDecoder",
    "NSGA2Config",
    "NSGA2RunResult",
    "NSGA2ScenarioResult",
    "ProblemLayout",
    "crowding_distance",
    "run_nsga2",
    "select_representatives",
    "solve_manifest_nsga2",
    "solve_scenario_nsga2",
]
