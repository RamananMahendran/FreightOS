"""Mesa agent = state holder for one train. SimPy owns the clock; this agent exposes a
SimPy process (`run`) that updates its state. Mesa's own step loop is deliberately unused."""
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Tuple
import mesa
import simpy
from .physics import SpeedProfile
from .track import TrackSegment


class Status(str, Enum):
    WAITING = "WAITING"
    LOADING = "LOADING"   # in yard: queued for a slot, loading, crew dwell
    QUEUED = "QUEUED"      # ready, waiting for predecessor / headway
    RUNNING = "RUNNING"
    ARRIVED = "ARRIVED"


class TrainAgent(mesa.Agent):
    def __init__(self, model: mesa.Model, train_id: str, ready_s: float, mass_kg: float,
                 segment: TrackSegment, profile: SpeedProfile, priority: float = 0.5,
                 load_tons: float = 0.0, unload_tons: float = 0.0,
                 yard=None, yard_arrival_s: float = 0.0,
                 headway_hold_s: Optional[float] = None,
                 closures=None):
        super().__init__(model)
        self.priority, self.load_tons, self.unload_tons = priority, load_tons, unload_tons
        self.yard, self.yard_arrival_s = yard, yard_arrival_s
        self.scheduled_ready_s = ready_s
        self.yard_arrive_s = self.load_start_s = self.load_end_s = None
        self.train_id, self.ready_s, self.mass_kg = train_id, ready_s, mass_kg
        self.segment, self.profile = segment, profile
        self.headway_hold_s = headway_hold_s
        self.closures = list(closures or [])
        self.status = Status.WAITING
        self.depart_s: Optional[float] = None
        self.arrive_s: Optional[float] = None
        self.position_m = 0.0
        self.speed_mps = 0.0
        self.log: List[Tuple[float, str]] = []

    def _note(self, env: simpy.Environment, msg: str):
        self.log.append((env.now, msg))

    def run(self, env: simpy.Environment, predecessor_departed: Optional[simpy.Event],
            departed_event: simpy.Event):
        """SimPy process: wait ready -> wait predecessor (sequence order) -> acquire entry
        lock (headway) -> follow speed profile -> arrive."""
        if self.yard is not None:
            self.status = Status.LOADING
            yield from self.yard.handle(self)          # sets self.ready_s
        else:
            yield env.timeout(self.ready_s)
        self.status = Status.QUEUED
        if predecessor_departed is not None:
            yield predecessor_departed
        for c in self.closures:
            if c.start_hrs * 3600.0 <= env.now < c.end_hrs * 3600.0:
                yield env.timeout(c.end_hrs * 3600.0 - env.now)
        req = self.segment.entry.request()
        yield req
        self.depart_s = env.now
        self.status = Status.RUNNING
        self._note(env, "depart")
        departed_event.succeed()
        env.process(self.segment.hold_entry(env, req, self.headway_hold_s))

        for ph in self.profile.phases:
            self.speed_mps = ph.v0
            yield env.timeout(ph.t1 - ph.t0)
            self.position_m, self.speed_mps = ph.s1, ph.v1
            self._note(env, ph.kind)
        self.status = Status.ARRIVED
        self.arrive_s = env.now
        self._note(env, "arrive")
