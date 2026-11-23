"""Train longitudinal physics: acceleration limited by tractive effort and power,
constant-deceleration braking, speed profile for one segment from rest to rest."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from .params import G, LocoParams


def braking_distance_m(v_mps: float, decel_mps2: float, delay_s: float = 0.0) -> float:
    """Distance to stop from speed v: travel during brake delay + v^2 / 2b."""
    return v_mps * delay_s + v_mps ** 2 / (2.0 * decel_mps2)


def accel_mps2(v_mps: float, mass_kg: float, loco: LocoParams) -> float:
    """Net acceleration at speed v: min(TE, P/v) minus rolling resistance, over mass."""
    force = min(loco.tractive_effort_n, loco.power_w / max(v_mps, 0.1))
    return (force - loco.rolling_resistance * mass_kg * G) / mass_kg


@dataclass(frozen=True)
class Phase:
    kind: str            # "accel" | "cruise" | "brake" | "stall" | "track"
    t0: float; t1: float  # s, relative to departure
    s0: float; s1: float  # m
    v0: float; v1: float  # m/s
    _ts: Optional[np.ndarray] = field(default=None, repr=False)
    _xs: Optional[np.ndarray] = field(default=None, repr=False)
    _vs: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class SpeedProfile:
    length_m: float
    phases: List[Phase]
    peak_speed_mps: float
    _acc_t: np.ndarray = field(repr=False, default=None)  # accel-phase samples for interpolation
    _acc_s: np.ndarray = field(repr=False, default=None)

    @property
    def total_time_s(self) -> float:
        return self.phases[-1].t1

    def position_at(self, t: float) -> float:
        """Distance travelled (m) at time t since departure."""
        t = min(max(t, 0.0), self.total_time_s)
        for ph in self.phases:
            if t <= ph.t1 or ph is self.phases[-1]:
                dt = t - ph.t0
                if ph.kind == "track" and ph._ts is not None and ph._xs is not None:
                    return float(np.interp(dt, ph._ts, ph._xs))
                if ph.kind == "stall":
                    return ph.s0
                if ph.kind == "accel":
                    if self._acc_t is not None and self._acc_s is not None and len(self._acc_t) > 1:
                        return float(np.interp(t, self._acc_t, self._acc_s))
                    return ph.s0 + 0.5 * (ph.v0 + ph.v1) * dt
                if ph.kind == "cruise":
                    return ph.s0 + ph.v0 * dt
                b = (ph.v0 - ph.v1) / (ph.t1 - ph.t0) if ph.t1 > ph.t0 else 0.0   # constant decel
                return ph.s0 + ph.v0 * dt - 0.5 * b * dt * dt
        return self.length_m

    def speed_at(self, t: float) -> float:
        t = min(max(t, 0.0), self.total_time_s)
        for ph in self.phases:
            if t <= ph.t1 or ph is self.phases[-1]:
                if ph.kind == "track" and ph._ts is not None and ph._vs is not None:
                    return float(np.interp(t - ph.t0, ph._ts, ph._vs))
                if ph.kind == "stall":
                    return 0.0
                if ph.t1 == ph.t0:
                    return ph.v1
                f = (t - ph.t0) / (ph.t1 - ph.t0)
                if ph.kind == "accel":
                    if self._acc_t is not None and len(self._acc_t) > 1:
                        i = int(np.searchsorted(self._acc_t, t))
                        i = min(max(i, 1), len(self._acc_t) - 1)
                        return float((self._acc_s[i] - self._acc_s[i - 1]) / max(self._acc_t[i] - self._acc_t[i - 1], 1e-9))
                    return ph.v0 + (ph.v1 - ph.v0) * f
                return ph.v0 + (ph.v1 - ph.v0) * f
        return 0.0

    def positions(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts)
        if ts.ndim == 0:
            return np.array(self.position_at(float(ts)))
        return np.array([self.position_at(float(t)) for t in ts.flat]).reshape(ts.shape)

    def speeds(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts)
        if ts.ndim == 0:
            return np.array(self.speed_at(float(ts)))
        return np.array([self.speed_at(float(t)) for t in ts.flat]).reshape(ts.shape)

    def time_at_position(self, x: float) -> float:
        """Inverse of position_at: returns time in seconds when train reaches distance x."""
        x = min(max(x, 0.0), self.length_m)
        for ph in self.phases:
            if ph.s0 <= x <= ph.s1 or ph is self.phases[-1]:
                dx = x - ph.s0
                if ph.kind == "track" and ph._xs is not None and ph._ts is not None:
                    return float(ph.t0 + np.interp(dx, ph._xs, ph._ts))
                if ph.kind == "stall":
                    return ph.t1
                if ph.kind == "accel":
                    if self._acc_s is not None and len(self._acc_s) > 1:
                        return float(np.interp(x, self._acc_s, self._acc_t))
                    v_avg = 0.5 * (ph.v0 + ph.v1)
                    return ph.t0 + (dx / max(v_avg, 1e-6))
                if ph.kind == "cruise":
                    return ph.t0 + (dx / max(ph.v0, 1e-6))
                b = (ph.v0 - ph.v1) / max(ph.t1 - ph.t0, 1e-9)
                if b > 1e-9:
                    v_sq = max(ph.v0 ** 2 - 2.0 * b * dx, 0.0)
                    v_curr = np.sqrt(v_sq)
                    return ph.t0 + (ph.v0 - v_curr) / b
                return ph.t0
        return self.total_time_s


def build_profile(length_m: float, v_limit_mps: float, mass_kg: float, loco: LocoParams,
                  decel_mps2: Optional[float] = None, dt: float = 0.5) -> SpeedProfile:
    """Rest-to-rest profile: accelerate -> cruise at v_limit -> brake to stop exactly at the end.
    Short segments never reach v_limit (accelerate, then brake)."""
    b = decel_mps2 or loco.service_decel_mps2
    t = s = v = 0.0
    ts, ss = [0.0], [0.0]
    brake_early = False
    while v < v_limit_mps - 1e-9:
        a = accel_mps2(v, mass_kg, loco)
        if a <= 0:
            raise ValueError(f"{loco.name} cannot accelerate {mass_kg/1000:.0f} t at {v:.1f} m/s")
        step = min(dt, (v_limit_mps - v) / a)
        v_new = min(v + a * step, v_limit_mps)
        s_new = s + 0.5 * (v + v_new) * step
        if s_new + v_new ** 2 / (2 * b) >= length_m:      # must start braking now
            brake_early = True
            break
        t, s, v = t + step, s_new, v_new
        ts.append(t); ss.append(s)

    phases: List[Phase] = []
    if len(ts) > 1:
        phases.append(Phase("accel", 0.0, t, 0.0, s, 0.0, v))
    if not brake_early:
        cruise_d = length_m - s - v ** 2 / (2 * b)
        if cruise_d > 1e-9:
            tc = cruise_d / v
            phases.append(Phase("cruise", t, t + tc, s, s + cruise_d, v, v))
            t, s = t + tc, s + cruise_d
    d_brake = length_m - s                                  # exact stop at segment end
    if d_brake > 1e-9 and v > 0:
        tb = 2.0 * d_brake / v                              # constant decel b' <= b
        phases.append(Phase("brake", t, t + tb, s, length_m, v, 0.0))
    return SpeedProfile(length_m, phases, v, np.array(ts), np.array(ss))


def with_breakdown(
    free: SpeedProfile,
    t_fail: float,
    duration_s: float,
    mass_kg: float,
    loco: LocoParams,
    v_limit_mps: float,
    dt: float = 0.5,
) -> SpeedProfile:
    """Builds a modified SpeedProfile incorporating a mid-corridor breakdown:
    Train operates freely up to t_fail, brakes to an emergency stop, stalls for
    duration_s, and then accelerates from rest to complete the journey to segment end.
    """
    t_fail = min(max(t_fail, 0.0), free.total_time_s)
    s_fail = free.position_at(t_fail)
    v_fail = free.speed_at(t_fail)

    b_em = loco.emergency_decel_mps2
    t_stop = v_fail / b_em if b_em > 0 else 0.0
    d_stop = (v_fail ** 2) / (2.0 * b_em) if b_em > 0 else 0.0
    s_stopped = min(s_fail + d_stop, free.length_m)

    new_phases: List[Phase] = []
    # 1. Free motion up to t_fail
    for ph in free.phases:
        if ph.t1 <= t_fail:
            new_phases.append(ph)
        elif ph.t0 < t_fail:
            dt_trim = t_fail - ph.t0
            new_phases.append(Phase(ph.kind, ph.t0, t_fail, ph.s0, s_fail, ph.v0, v_fail))
            break
        else:
            break

    # 2. Emergency braking phase to 0
    t_stop_end = t_fail + t_stop
    if t_stop > 1e-6:
        new_phases.append(Phase("brake", t_fail, t_stop_end, s_fail, s_stopped, v_fail, 0.0))
    else:
        t_stop_end = t_fail

    # 3. Stall phase at standstill
    t_resume = t_stop_end + duration_s
    new_phases.append(Phase("stall", t_stop_end, t_resume, s_stopped, s_stopped, 0.0, 0.0))

    # 4. Resume to segment end
    d_rem = max(0.0, free.length_m - s_stopped)
    if d_rem > 1e-6:
        rem_profile = build_profile(d_rem, v_limit_mps, mass_kg, loco, dt=dt)
        for rph in rem_profile.phases:
            new_phases.append(
                Phase(
                    rph.kind,
                    rph.t0 + t_resume,
                    rph.t1 + t_resume,
                    rph.s0 + s_stopped,
                    rph.s1 + s_stopped,
                    rph.v0,
                    rph.v1,
                )
            )

    return SpeedProfile(free.length_m, new_phases, free.peak_speed_mps)

