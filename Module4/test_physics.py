import pytest
from hypothesis import given, settings, strategies as st
from Module4.params import IDEAL, LOCOS
from Module4.physics import braking_distance_m, accel_mps2, build_profile

WAG9 = LOCOS["WAG_9"]

def test_braking_distance_formula():
    assert braking_distance_m(20, 0.5) == pytest.approx(400.0)
    assert braking_distance_m(20, 0.5, delay_s=5) == pytest.approx(500.0)

def test_accel_power_limited_at_speed():
    m = 4.0e6
    assert accel_mps2(1.0, m, WAG9) > accel_mps2(20.0, m, WAG9) > 0

def test_heavier_train_is_slower():
    L, v = 176_000, 80 / 3.6
    assert build_profile(L, v, 6e6, WAG9).total_time_s > build_profile(L, v, 3e6, WAG9).total_time_s

def test_ideal_loco_matches_free_flow():
    L, v = 176_000, 80 / 3.6
    assert build_profile(L, v, 4e6, IDEAL).total_time_s == pytest.approx(L / v, abs=1.0)

def test_short_segment_never_reaches_limit():
    p = build_profile(3_000, 80 / 3.6, 4e6, WAG9)
    assert p.peak_speed_mps < 80 / 3.6 and [ph.kind for ph in p.phases] == ["accel", "brake"]

def test_position_monotone_and_ends_at_length():
    p = build_profile(50_000, 80 / 3.6, 4e6, WAG9)
    xs = [p.position_at(t) for t in range(0, int(p.total_time_s) + 1, 30)]
    assert all(b >= a - 1e-6 for a, b in zip(xs, xs[1:]))
    assert p.position_at(p.total_time_s) == pytest.approx(50_000, abs=1.0)

def test_underpowered_loco_raises():
    with pytest.raises(ValueError):
        build_profile(10_000, 22.0, 1e9, WAG9)

@settings(max_examples=60, deadline=None)
@given(L=st.floats(500, 300_000), mass=st.floats(5e5, 8e6), vkmh=st.floats(20, 100))
def test_property_distance_conserved(L, mass, vkmh):
    p = build_profile(L, vkmh / 3.6, mass, WAG9)
    assert sum(ph.s1 - ph.s0 for ph in p.phases) == pytest.approx(L, rel=1e-6)
    assert p.total_time_s >= L / (vkmh / 3.6) - 1e-6      # physics can't beat free-flow
    assert p.peak_speed_mps <= vkmh / 3.6 + 1e-9
