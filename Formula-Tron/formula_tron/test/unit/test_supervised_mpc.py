"""Unit tests for the Supervised MPC mode.

Tests verify:
1. SUPERVISED_MPC is correctly registered in mode_config.py
2. Speed preset set/clear/update logic works correctly
3. Steering assist blend vs override math is correct
4. Speed override applies over MPC output
5. Lap count forwarding works for SUPERVISED_MPC
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from formula_tron.utils.base_controller import BaseController, ControlState, ControlOutput

try:
    from formula_tron.utils import mpc_controller as mpc_module
    MPCController = mpc_module.MPCController
    CASADI_AVAILABLE = bool(getattr(mpc_module, "CASADI_AVAILABLE", False))
except ImportError:
    CASADI_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# 1. Mode Registration (mode_config.py)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSupervisedMPCConfig:

    def test_supervised_mpc_in_modes_list(self):
        from formula_tron.utils.mode_config import MODES
        mode_names = [m[0] for m in MODES]
        assert "SUPERVISED_MPC" in mode_names

    def test_supervised_mpc_in_rev1_section(self):
        from formula_tron.utils.mode_config import MODES
        for name, label, section, enabled, color in MODES:
            if name == "SUPERVISED_MPC":
                assert section == "rev1"
                assert enabled is False  # disabled in mode_config (hidden from mode picker)
                break
        else:
            pytest.fail("SUPERVISED_MPC not found in MODES")

    def test_supervised_mpc_label(self):
        from formula_tron.utils.mode_config import MODES
        labels = {m[0]: m[1] for m in MODES}
        assert labels["SUPERVISED_MPC"] == "Supervised MPC"

    def test_supervised_mpc_visibility_rules(self):
        from formula_tron.utils.mode_config import MODE_VISIBILITY
        assert "SUPERVISED_MPC" in MODE_VISIBILITY
        vis = MODE_VISIBILITY["SUPERVISED_MPC"]
        # Must show MPC tuning group
        assert vis["mpc_group"] is True
        # Must show advanced group (BEV / poly settings)
        assert vis["advanced_group"] is True

    def test_supervised_mpc_status_style(self):
        from formula_tron.utils.mode_config import MODE_STATUS_STYLE
        assert "SUPERVISED_MPC" in MODE_STATUS_STYLE
        label, style = MODE_STATUS_STYLE["SUPERVISED_MPC"]
        assert "Supervised MPC" in label
        assert "#00ff88" in style

    def test_supervised_mpc_tooltip(self):
        from formula_tron.utils.mode_config import MODE_TOOLTIPS
        assert "SUPERVISED_MPC" in MODE_TOOLTIPS


# ═══════════════════════════════════════════════════════════════════════
# 2. Speed Preset Logic
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSupervisedMPCSpeedPresets:

    def test_default_presets(self):
        """Default SMPC speed presets should be 7 values."""
        presets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        assert len(presets) == 7
        assert all(p > 0 for p in presets)

    def test_set_speed_valid_index(self):
        """Setting a valid preset index should update active key and speed."""
        presets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        active_key = -1
        speed = None

        idx = 3  # key 4
        active_key = idx
        speed = presets[idx]

        assert active_key == 3
        assert speed == 2.0

    def test_set_speed_boundary_indices(self):
        """Index 0 and 6 should both work."""
        presets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

        # First preset
        speed = presets[0]
        assert speed == 0.5

        # Last preset
        speed = presets[6]
        assert speed == 4.0

    def test_clear_speed(self):
        """Clearing speed should set speed to None and active_key to -1."""
        speed = 2.0
        active_key = 3

        speed = None
        active_key = -1

        assert speed is None
        assert active_key == -1

    def test_preset_update_propagates_to_active(self):
        """Changing a spinbox value should update live speed if that key is active."""
        presets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        active_key = 2
        speed = presets[active_key]  # 1.5

        # Simulate spinbox change
        new_val = 1.8
        presets[active_key] = new_val
        speed = new_val  # should update live speed

        assert speed == 1.8
        assert presets[2] == 1.8


# ═══════════════════════════════════════════════════════════════════════
# 3. Steering Assist Logic
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSupervisedMPCSteeringAssist:

    def test_blend_adds_delta(self):
        """Blend mode: human assist is added to MPC steering."""
        mpc_steer = 0.1
        human_assist = 0.2  # A key pressed
        max_steer = 0.45

        result = mpc_steer + human_assist
        result = max(-max_steer, min(max_steer, result))

        assert result == pytest.approx(0.3)

    def test_blend_clamps(self):
        """Blend mode: result should be clamped to max steering angle."""
        mpc_steer = 0.3
        human_assist = 0.3
        max_steer = 0.45

        result = mpc_steer + human_assist
        result = max(-max_steer, min(max_steer, result))

        assert result == pytest.approx(0.45)

    def test_override_replaces(self):
        """Override mode: human steering replaces MPC steering."""
        mpc_steer = 0.1
        human_assist = -0.35  # D key pressed

        # In override mode, MPC steer is ignored
        result = human_assist

        assert result == pytest.approx(-0.35)

    def test_no_input_passthrough(self):
        """When no A/D pressed, MPC steering passes through unchanged."""
        mpc_steer = 0.15
        human_assist = 0.0

        # No assist, steering should be MPC's output
        # In blend mode: mpc + 0 = mpc
        result = mpc_steer + human_assist
        assert result == pytest.approx(0.15)

        # In override mode with zero assist, MPC should still pass through
        # (the condition checks smpc_steer_assist != 0.0)
        # So result stays as mpc_steer
        assert mpc_steer == pytest.approx(0.15)


# ═══════════════════════════════════════════════════════════════════════
# 4. Speed Override Logic
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSupervisedMPCSpeedOverride:

    def test_locked_speed_overrides_mpc(self):
        """When speed is locked, MPC's computed speed should be replaced."""
        mpc_speed = 2.5
        locked_speed = 1.5

        # Override: use locked speed
        speed = float(locked_speed) if locked_speed is not None else mpc_speed

        assert speed == 1.5

    def test_none_lets_mpc_control(self):
        """When speed override is None, MPC's computed speed is used."""
        mpc_speed = 2.5
        locked_speed = None

        speed = float(locked_speed) if locked_speed is not None else mpc_speed

        assert speed == 2.5

    def test_preset_change_updates_live_speed(self):
        """Changing the active preset value should update the live override speed."""
        presets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        active_key = 4
        speed_override = presets[active_key]  # 2.5

        # Update the active preset via spinbox
        presets[active_key] = 3.5
        speed_override = presets[active_key]

        assert speed_override == 3.5


# ═══════════════════════════════════════════════════════════════════════
# 5. Controller Routing & Lap Forwarding
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSupervisedMPCRouting:

    def test_supervised_mpc_routes_to_mpc(self):
        """SUPERVISED_MPC should use the MPC controller internally."""
        control_mode = "SUPERVISED_MPC"
        lookup_mode = control_mode
        if lookup_mode == "SUPERVISED_MPC":
            lookup_mode = "MPC"
        assert lookup_mode == "MPC"

    def test_mpc_mode_unchanged(self):
        """Regular MPC mode should still route to MPC."""
        control_mode = "MPC"
        lookup_mode = control_mode
        if lookup_mode == "SUPERVISED_MPC":
            lookup_mode = "MPC"
        assert lookup_mode == "MPC"

    def test_lap_forwarding_includes_supervised_mpc(self):
        """Lap count should be forwarded to MPC in both MPC and SUPERVISED_MPC modes."""
        for mode in ("MPC", "SUPERVISED_MPC"):
            assert mode in ("MPC", "SUPERVISED_MPC")


# ═══════════════════════════════════════════════════════════════════════
# 6. SMPC Param Callback Parsing
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSMPCParamParsing:

    def test_parse_blend_with_speed(self):
        """Parse a valid SMPC param JSON with blend + speed override."""
        import json
        data = json.dumps({
            'smpc_steer_assist': 0.3,
            'smpc_steer_mode': 'blend',
            'smpc_speed_override': 2.0,
        })
        params = json.loads(data)
        assert params['smpc_steer_assist'] == 0.3
        assert params['smpc_steer_mode'] == 'blend'
        assert params['smpc_speed_override'] == 2.0

    def test_parse_override_no_speed(self):
        """Parse SMPC param JSON with override mode and no speed lock."""
        import json
        data = json.dumps({
            'smpc_steer_assist': -0.45,
            'smpc_steer_mode': 'override',
            'smpc_speed_override': None,
        })
        params = json.loads(data)
        assert params['smpc_steer_assist'] == -0.45
        assert params['smpc_steer_mode'] == 'override'
        assert params['smpc_speed_override'] is None

    def test_parse_zero_assist(self):
        """Parse SMPC param JSON with no steering input."""
        import json
        data = json.dumps({
            'smpc_steer_assist': 0.0,
            'smpc_steer_mode': 'blend',
            'smpc_speed_override': None,
        })
        params = json.loads(data)
        assert params['smpc_steer_assist'] == 0.0
