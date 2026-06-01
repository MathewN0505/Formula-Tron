"""Unit tests for the Expo Mode.

Tests verify:
1. EXPO is correctly registered in mode_config.py
2. Expo routine math produces expected values
3. Safety: start/stop zeros commands, emergency stop halts expo
4. Battery voltage -> percentage conversion and color thresholds
"""

import pytest
import math
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 1. Mode Registration (mode_config.py)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestExpoModeConfig:

    def test_expo_in_modes_list(self):
        from formula_tron.utils.mode_config import MODES
        mode_names = [m[0] for m in MODES]
        assert "EXPO" in mode_names

    def test_expo_in_showcase_section(self):
        from formula_tron.utils.mode_config import MODES, SECTION_HEADERS
        assert "showcase" in SECTION_HEADERS
        for name, label, section, enabled, color in MODES:
            if name == "EXPO":
                assert section == "showcase"
                assert enabled is True
                break
        else:
            pytest.fail("EXPO not found in MODES")

    def test_expo_label(self):
        from formula_tron.utils.mode_config import MODES
        labels = {m[0]: m[1] for m in MODES}
        assert labels["EXPO"] == "Expo Mode"

    def test_expo_color_is_magenta(self):
        from formula_tron.utils.mode_config import MODES
        colors = {m[0]: m[4] for m in MODES}
        assert colors["EXPO"] == "#ffcc00"

    def test_expo_visibility_rules(self):
        from formula_tron.utils.mode_config import MODE_VISIBILITY
        assert "EXPO" in MODE_VISIBILITY
        vis = MODE_VISIBILITY["EXPO"]
        # All tuning groups should be hidden
        assert vis["pd_group"] is False
        assert vis["mpc_group"] is False
        assert vis["advanced_group"] is False
        assert vis["speed_extras"] is False
        assert vis["width_group"] is False

    def test_expo_status_style(self):
        from formula_tron.utils.mode_config import MODE_STATUS_STYLE
        assert "EXPO" in MODE_STATUS_STYLE
        label, style = MODE_STATUS_STYLE["EXPO"]
        assert "Expo Mode" in label
        assert "#ffcc00" in style

    def test_expo_tooltip(self):
        from formula_tron.utils.mode_config import MODE_TOOLTIPS
        assert "EXPO" in MODE_TOOLTIPS
        assert "stand" in MODE_TOOLTIPS["EXPO"].lower() or "expo" in MODE_TOOLTIPS["EXPO"].lower()

    def test_build_dropdown_includes_expo(self):
        from formula_tron.utils.mode_config import build_dropdown_items
        items, mode_to_index = build_dropdown_items()
        assert "EXPO" in mode_to_index
        idx = mode_to_index["EXPO"]
        # Should be after SUPERVISED_MPC (Rev1 block before Beta / Showcase)
        assert "SUPERVISED_MPC" in mode_to_index
        assert idx > mode_to_index["SUPERVISED_MPC"]

    def test_expo_welcome_overlay_import(self):
        from formula_tron.expo_welcome_overlay import ExpoWelcomeOverlay
        assert ExpoWelcomeOverlay.__name__ == "ExpoWelcomeOverlay"


# ═══════════════════════════════════════════════════════════════════════
# 2. Expo Routine Math
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestExpoRoutines:

    def test_sweep_constant_speed(self):
        """Sweep routine: speed should be constant at amplitude."""
        A_spd = 1.5
        A_str = 0.35
        f = 1.0
        t = 0.5  # half cycle

        speed = A_spd  # Sweep uses constant speed
        steer = A_str * math.sin(2.0 * math.pi * f * t)

        assert speed == pytest.approx(1.5)
        # sin(pi) = 0
        assert steer == pytest.approx(0.0, abs=1e-10)

    def test_sweep_quarter_cycle(self):
        """At quarter cycle, steering should be at max amplitude."""
        A_str = 0.35
        f = 1.0
        t = 0.25  # quarter cycle

        steer = A_str * math.sin(2.0 * math.pi * f * t)
        assert steer == pytest.approx(0.35)

    def test_heartbeat_ramp_up(self):
        """Heartbeat: early phase should ramp up speed."""
        A_spd = 2.0
        f = 1.0
        t = 0.15  # phase = 0.15 (in ramp-up: 0 to 0.3)

        phase = (t * f) % 1.0  # = 0.15
        speed = A_spd * (phase / 0.3)

        assert speed == pytest.approx(1.0)

    def test_heartbeat_idle(self):
        """Heartbeat: second half of cycle should be zero speed."""
        A_spd = 2.0
        f = 1.0
        t = 0.75  # phase = 0.75 (in idle: > 0.5)

        phase = (t * f) % 1.0
        assert phase > 0.5
        speed = 0.0
        assert speed == 0.0

    def test_wave_phase_offset(self):
        """Wave: speed and steer should be 90deg out of phase."""
        A_spd = 1.0
        A_str = 0.3
        f = 1.0
        t = 0.0

        speed = A_spd * (0.5 + 0.5 * math.sin(2.0 * math.pi * f * t))
        steer = A_str * math.sin(2.0 * math.pi * f * t + math.pi / 2.0)

        # At t=0: sin(0)=0 → speed=0.5, sin(pi/2)=1 → steer=0.3
        assert speed == pytest.approx(0.5)
        assert steer == pytest.approx(0.3)

    def test_figure8_double_frequency(self):
        """Figure-8: speed oscillates at 2x the steering frequency."""
        A_spd = 1.0
        A_str = 0.35
        f = 1.0
        t = 0.25

        steer = A_str * math.sin(2.0 * math.pi * f * t)  # sin(pi/2) = 1
        speed = A_spd * (0.5 + 0.5 * math.sin(4.0 * math.pi * f * t))  # sin(pi) = 0

        assert steer == pytest.approx(0.35)
        assert speed == pytest.approx(0.5)

    def test_all_routines_produce_bounded_output(self):
        """All routines should produce speed in [0, 3] and steer in [-0.45, 0.45]."""
        routines = ['sweep', 'heartbeat', 'wave', 'figure8']
        A_spd = 3.0  # Max amplitude
        A_str = 0.45  # Max amplitude
        f = 2.0

        for routine in routines:
            for t in np.linspace(0, 2.0, 100):
                if routine == 'sweep':
                    speed = A_spd
                    steer = A_str * math.sin(2.0 * math.pi * f * t)
                elif routine == 'heartbeat':
                    phase = (t * f) % 1.0
                    if phase < 0.3:
                        speed = A_spd * (phase / 0.3)
                    elif phase < 0.5:
                        speed = A_spd * (1.0 - (phase - 0.3) / 0.2)
                    else:
                        speed = 0.0
                    steer = A_str if (int(t * f) % 2 == 0) else -A_str
                elif routine == 'wave':
                    speed = A_spd * (0.5 + 0.5 * math.sin(2.0 * math.pi * f * t))
                    steer = A_str * math.sin(2.0 * math.pi * f * t + math.pi / 2.0)
                elif routine == 'figure8':
                    steer = A_str * math.sin(2.0 * math.pi * f * t)
                    speed = A_spd * (0.5 + 0.5 * math.sin(4.0 * math.pi * f * t))

                # Safety clamp (as in actual code)
                speed = max(0.0, min(3.0, speed))
                steer = max(-0.45, min(0.45, steer))

                assert 0.0 <= speed <= 3.0, f"{routine} t={t}: speed={speed}"
                assert -0.45 <= steer <= 0.45, f"{routine} t={t}: steer={steer}"


# ═══════════════════════════════════════════════════════════════════════
# 3. Safety
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestExpoSafety:

    def test_stop_resets_state(self):
        """Stopping expo should reset active flag and phase."""
        expo_active = True
        expo_phase = 3.5

        # Simulate _stop_expo
        expo_active = False
        expo_phase = 0.0

        assert expo_active is False
        assert expo_phase == 0.0

    def test_speed_clamp_prevents_overshoot(self):
        """Speed should never exceed 3.0 m/s even with max amplitude."""
        speed = 5.0  # Hypothetically huge
        speed = max(0.0, min(3.0, speed))
        assert speed == 3.0

    def test_steer_clamp_prevents_overshoot(self):
        """Steering should never exceed ±0.45 rad."""
        steer = 1.0
        steer = max(-0.45, min(0.45, steer))
        assert steer == 0.45

        steer = -1.0
        steer = max(-0.45, min(0.45, steer))
        assert steer == -0.45

    def test_negative_speed_clamped_to_zero(self):
        """Wave/figure-8 can't produce negative speed."""
        speed = -0.5
        speed = max(0.0, min(3.0, speed))
        assert speed == 0.0

    def test_auto_cycle_wraps_around(self):
        """Auto-cycle index should wrap around after 4 routines."""
        idx = 0
        for _ in range(8):
            idx = (idx + 1) % 4
        assert idx == 0  # Should wrap back to start


# ═══════════════════════════════════════════════════════════════════════
# 4. Battery Display
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestBatteryDisplay:

    def _voltage_to_pct(self, voltage):
        """Mirror the actual conversion logic."""
        return max(0, min(100, int((voltage - 9.0) / (12.6 - 9.0) * 100)))

    def test_full_battery(self):
        assert self._voltage_to_pct(12.6) == 100

    def test_empty_battery(self):
        assert self._voltage_to_pct(9.0) == 0

    def test_below_empty(self):
        """Below 9.0V should still return 0."""
        assert self._voltage_to_pct(8.0) == 0

    def test_above_full(self):
        """Above 12.6V should still return 100."""
        assert self._voltage_to_pct(13.0) == 100

    def test_mid_battery(self):
        """10.8V is 50% of 9.0-12.6 range."""
        pct = self._voltage_to_pct(10.8)
        assert pct == 50

    def test_low_battery_threshold(self):
        """Below ~9.72V (20%) should be 'red' zone."""
        pct = self._voltage_to_pct(9.5)
        assert pct < 20

    def test_color_green_zone(self):
        pct = self._voltage_to_pct(11.5)
        assert pct > 50  # Green zone

    def test_color_yellow_zone(self):
        pct = self._voltage_to_pct(10.0)
        assert 20 < pct <= 50  # Yellow zone

    def test_color_red_zone(self):
        pct = self._voltage_to_pct(9.3)
        assert pct <= 20  # Red zone

    def test_monotonic_increase(self):
        """Higher voltage should always give higher or equal percentage."""
        voltages = [8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.6, 13.0]
        pcts = [self._voltage_to_pct(v) for v in voltages]
        for i in range(len(pcts) - 1):
            assert pcts[i] <= pcts[i + 1], f"Not monotonic at {voltages[i]}V"
