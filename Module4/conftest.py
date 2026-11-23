import os, sys, pytest
from pathlib import Path
from Module4.schemas import Plan

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

M3 = os.environ.get("MODULE3_PATH", ROOT)
if os.path.isdir(M3) and M3 not in sys.path:
    sys.path.insert(0, M3)

SEGMENTS = [{"id": "CORRIDOR_MAIN", "traversing_time_hrs": 2.0, "safe_headway_hrs": 0.25}]

def make_plan(**over) -> Plan:
    trains = [
        {"id": "T1", "origin": "A", "destination": "B", "ready_time_hrs": 0.0, "max_capacity_tons": 2600, "priority": 0.8},
        {"id": "T2", "origin": "A", "destination": "B", "ready_time_hrs": 0.1, "max_capacity_tons": 2800, "priority": 0.5},
    ]
    cargos = [
        {"id": "C1", "origin": "A", "destination": "B", "weight": 2000, "priority": 0.8, "due_date_offset_hrs": 3.0},
        {"id": "C2", "origin": "A", "destination": "B", "weight": 2000, "priority": 0.5, "due_date_offset_hrs": 3.0},
    ]
    d = dict(trains=trains, cargos=cargos, segments=SEGMENTS, train_sequence=["T1", "T2"],
             allocation={"C1": "T1", "C2": "T2"},
             remaining_capacity={"T1": 600, "T2": 800},
             claimed_arrivals={"T1": 2.0, "T2": 2.25})   # T2 departs at max(0.1, 0.25)=0.25
    d.update(over)
    return Plan(**d)

@pytest.fixture
def plan():
    return make_plan()
