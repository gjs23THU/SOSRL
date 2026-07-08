import json
import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from os import PathLike

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(path: str | PathLike[str] = CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


CONFIG = load_config()

func_type2idx = {func_type: (idx + 1) / len(CONFIG.get("funcs", {})) for idx, func_type in enumerate(CONFIG.get("funcs", {}))}

@dataclass
class ComponentSystem:
    index:int
    name:str
    cost:int
    available_from:int
    available_until:int
    func_type:str

@dataclass
class Operation:
    index:int
    name:str
    func_type:str
    duration:int
    release_time:int
    alocated_sys: int | None = None
    start_time: int | None = None
    
    def __init__(self, index:int, name:str, func_type:str, duration:int, release_time:int=0):
        self.index = index
        self.name = name
        self.func_type = func_type
        self.duration = duration
        self.release_time = release_time

@dataclass
class Task:
    index:int
    name:str
    operations:list[Operation]
    def __init__(self, index:int, name:str, operations:list[Operation]):
        self.index = index
        self.name = name
        self.operations = operations
    
    def randomize_operations(self, func_types:dict[str,int], op_duration:tuple[int, int], op_per_task:int):
        self.operations = []
        rel_time = 0
        for op_index in range(op_per_task):
            func_type = random.choices(list(func_types.keys()), weights=list(func_types.values()))[0]
            func_type = func_type2idx.get(func_type, 0)  # Convert func_type to its corresponding index
            duration = random.randint(op_duration[0], op_duration[1])
            operation = Operation(
                index=op_index,
                name=f"Op_{self.index}_{op_index}",
                func_type=func_type,
                duration=duration,
                release_time=rel_time
            )
            rel_time += duration  # Update release time for the next operation
            self.operations.append(operation)

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
                func_type=func_type2idx.get(func_type, 0),
            )
        )
    return tuple(sos)

FULL_SOS = build_sos_from_config(CONFIG)

def random_select_sos(selected_num:int, config: dict[str, Any]=CONFIG) -> tuple[ComponentSystem]:
    selected_sos: list[ComponentSystem] = []
    for key in config.get("funcs", {}):
        selected_sos.append(random.sample([s for s in FULL_SOS if s.func_type == func_type2idx.get(key, 0)], 1)[0])
    selected_sos.extend(random.sample([s for s in FULL_SOS if s not in selected_sos], selected_num-len(selected_sos)))
    return tuple(selected_sos)

def build_mission_from_config(config:dict[str,Any]):
    mission = []
    for task_index in range(config.get("total_task", 30)):
        task = Task(
            index=task_index,
            name=f"Task_{task_index}",
            operations=[]
        )
        task.randomize_operations(
            func_types=config.get("funcs", {}),
            op_duration=config.get("op_duration", (20, 40)),
            op_per_task=config.get("op_per_task", 4)
        )
        mission.append(task)
    return mission
