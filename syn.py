import json
import random
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(path: str | PathLike[str] = CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


CONFIG = load_config()

func_type2idx = {
    func_type: idx
    for idx, func_type in enumerate(CONFIG.get("funcs", {}))
}


@dataclass
class ComponentSystem:
    index: int
    name: str
    cost: int
    available_from: int
    available_until: int
    func_type: int


@dataclass
class Operation:
    index: int
    name: str
    func_type: int
    duration: int
    release_time: int = 0


@dataclass
class Task:
    index: int
    name: str
    operations: list[Operation]
    release_time: int = 0
    due_time: int = 0

    def randomize_operations(
        self,
        func_types: dict[str, int],
        op_duration: tuple[int, int],
        op_per_task: int,
    ):
        self.operations = []
        release_time = self.release_time
        for op_index in range(op_per_task):
            func_type = random.choices(list(func_types.keys()), weights=list(func_types.values()))[0]
            func_type = func_type2idx[func_type]
            duration = random.randint(op_duration[0], op_duration[1])
            operation = Operation(
                index=op_index,
                name=f"Op_{self.index}_{op_index}",
                func_type=func_type,
                duration=duration,
                release_time=release_time,
            )
            release_time += duration
            self.operations.append(operation)

    def set_due_time(self, tightness: float = 3.0):
        total_duration = sum(op.duration for op in self.operations)
        self.due_time = self.release_time + int(total_duration * tightness)

def build_sos_from_config(config: dict[str, Any] | str | PathLike[str]) -> tuple[ComponentSystem, ...]:
    if isinstance(config, (str, PathLike)):
        config = load_config(config)

    sos: list[ComponentSystem] = []
    capability_map = config.get("funcs")
    candidate_systems = config.get("candidate_systems")
    for index, item in enumerate(candidate_systems):
        func_type = item["function_type"]
        if func_type not in capability_map:
            raise ValueError(f"candidate_systems[{index}].function_type '{func_type}' is not defined in function_type_capabilities")
        sos.append(
            ComponentSystem(
                index=index,
                name=item["name"],
                cost=int(item["cost"]),
                available_from=int(item["available_from"]),
                available_until=int(item["available_until"]),
                func_type=func_type2idx[func_type],
            )
        )
    return tuple(sos)

FULL_SOS = build_sos_from_config(CONFIG)

def random_select_sos(selected_num: int) -> tuple[ComponentSystem, ...]:
    selected_sos: list[ComponentSystem] = []
    for key in CONFIG.get("funcs", {}):
        candidates = [system for system in FULL_SOS if system.func_type == func_type2idx[key]]
        selected_sos.append(random.choice(candidates))
    remaining = [system for system in FULL_SOS if system not in selected_sos]
    selected_sos.extend(random.sample(remaining, selected_num - len(selected_sos)))
    return tuple(selected_sos)


def build_mission_from_config(config: dict[str, Any]) -> list[Task]:
    tightness_upper = float(config.get("due_time_tightness", 3.0))
    if tightness_upper < 1.0:
        raise ValueError("due_time_tightness must be at least 1.0.")

    mission = []
    for task_index in range(config.get("total_task", 30)):
        task = Task(
            index=task_index,
            name=f"Task_{task_index}",
            operations=[],
            release_time=0
        )
        task.randomize_operations(
            func_types=config.get("funcs", {}),
            op_duration=config.get("op_duration", (20, 40)),
            op_per_task=config.get("op_per_task", 4)
        )
        task.set_due_time(tightness=random.uniform(1.0, tightness_upper))
        mission.append(task)
    return mission
