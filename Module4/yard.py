"""Yard workflow (SimPy): trains queue for a loading slot, load, then complete crew dwell.
Slots are a PriorityResource so higher-priority trains are loaded first (ties FIFO)."""
from __future__ import annotations
from dataclasses import dataclass
import simpy
from .params import YardParams

HRS = 3600.0


@dataclass
class Yard:
    env: simpy.Environment
    station: str
    params: YardParams
    busy_s: float = 0.0
    loads_done: int = 0

    def __post_init__(self):
        self.slots = simpy.PriorityResource(self.env, capacity=self.params.slots)

    def handle(self, agent):
        """SimPy process for one train. Sets agent.ready_s (when it may depart) and records timings."""
        env, p = self.env, self.params
        yield env.timeout(max(0.0, agent.yard_arrival_s - env.now))
        agent.yard_arrive_s = env.now
        work_s = (p.unload_hrs(agent.unload_tons) + p.load_hrs(agent.load_tons)) * HRS
        crew_s = p.min_crew_dwell_hrs * HRS
        crew_done = env.now + crew_s                       # used only if crew_concurrent

        with self.slots.request(priority=-agent.priority) as req:   # lower value = served first
            yield req
            agent.load_start_s = env.now
            yield env.timeout(work_s)
            agent.load_end_s = env.now
        self.busy_s += work_s
        self.loads_done += 1

        if p.crew_concurrent:
            yield env.timeout(max(0.0, crew_done - env.now))
        else:
            yield env.timeout(crew_s)                      # spec 5.1: Ul + Lo + dw in sequence
        agent.ready_s = env.now
