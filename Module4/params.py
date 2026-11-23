"""Simulation parameters. EVERYTHING HERE IS AN ASSUMPTION, not measured IR data.

Order-of-magnitude values from public locomotive specs; treat sim results as
'plausible physics', not validated against real operations (see spec Section 11 discipline).
Replace with RDSO/IR figures when available.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

G = 9.81  # m/s^2


@dataclass(frozen=True)
class LocoParams:
    name: str
    power_w: float                 # continuous traction power at rail      [ASSUMED]
    tractive_effort_n: float       # starting tractive effort                [ASSUMED]
    service_decel_mps2: float      # service brake, loaded freight           [ASSUMED]
    emergency_decel_mps2: float    # emergency brake                         [ASSUMED]
    brake_delay_s: float           # brake-pipe propagation / build-up delay [ASSUMED]
    rolling_resistance: float = 0.003   # dimensionless (N per N of weight)  [ASSUMED]


LOCOS: Dict[str, LocoParams] = {
    "WAG_9":  LocoParams("WAG_9",  6.0e6, 460e3, 0.35, 0.60, 8.0),
    "WAG_12B": LocoParams("WAG_12B", 9.0e6, 700e3, 0.35, 0.60, 8.0),
    "WDG_4D": LocoParams("WDG_4D", 3.3e6, 540e3, 0.30, 0.55, 8.0),
}

# Idealised loco: instant accel/brake. A physics-free sim must reproduce Module 3 exactly;
# used for calibration only.
IDEAL = LocoParams("IDEAL", 1e15, 1e15, 1e6, 1e6, 0.0, rolling_resistance=0.0)

DEFAULT_LOCO = "WAG_9"
TARE_FRACTION_OF_CAPACITY = 0.30   # empty-wagon mass as a fraction of payload capacity [ASSUMED]


@dataclass(frozen=True)
class YardParams:
    """Yard handling at a train's ORIGIN station.   ALL VALUES ARE ASSUMPTIONS (not IR data).

    A 'loading slot' = one loading bay served by one gantry crane / loader. It is the unit of
    yard capacity: at most `slots` trains are loaded at once, each at `loading_rate_tph`.
    Dwell follows spec 5.1:  d - a  >=  Ul + Lo + dw   (unloading + loading + crew dwell, in sequence)
    unless `crew_concurrent=True` (crew change overlaps with loading: dwell = max(Ul+Lo, dw)).
    """
    slots: int = 2                       # parallel loading bays per yard             [ASSUMED]
    loading_rate_tph: float = 800.0      # tonnes/hour per slot                       [ASSUMED - user-provided e.g.]
    unloading_rate_tph: float = 800.0    # tonnes/hour per slot                       [ASSUMED]
    min_crew_dwell_hrs: float = 0.5      # minimum crew dwell dw (30 min)             [ASSUMED - user-provided e.g.]
    crew_concurrent: bool = False        # False = spec-literal (sequential)

    @classmethod
    def instant(cls) -> "YardParams":
        """Zero-time, unlimited yard: sim must then reproduce Module 3 exactly (calibration)."""
        return cls(slots=10**6, loading_rate_tph=float("inf"), unloading_rate_tph=float("inf"),
                   min_crew_dwell_hrs=0.0)

    def load_hrs(self, tons: float) -> float:
        return 0.0 if tons <= 0 else tons / self.loading_rate_tph

    def unload_hrs(self, tons: float) -> float:
        return 0.0 if tons <= 0 else tons / self.unloading_rate_tph

    def min_dwell_hrs(self, load_tons: float, unload_tons: float = 0.0) -> float:
        lo, ul, dw = self.load_hrs(load_tons), self.unload_hrs(unload_tons), self.min_crew_dwell_hrs
        return max(lo + ul, dw) if self.crew_concurrent else lo + ul + dw
