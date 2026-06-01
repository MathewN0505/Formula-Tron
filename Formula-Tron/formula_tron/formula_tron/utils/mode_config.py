"""
Mode configuration for the Formula-Tron control GUI.

Central definition of all control modes, their display properties,
and dropdown organization. Consumed by control_gui (mode overlay and visibility).

Modes listed in MODES populate the on-screen picker; additional mode strings may
still be selected via ROS (/tuning/control_mode) or presets — see control_gui.

To add a new mode:
1. Add an entry to MODES list
2. Create the controller class extending BaseController
3. Register it in vision_controller.py's __init__
"""

MODES = [
    # Rev0 — Proven algorithms
    ("POLY_LOOKAHEAD", "Poly LookAhead", "rev0", True,  None),
    ("LEGACY",         "LEGACY",         "rev0", True,  None),
    
    # Rev1 — Stable New Algorithms
    ("DISCRETE",       "Discrete Speed",  "rev1", False, "#ffcc00"),
    ("MPC",            "MPC",             "rev1", True,  None),
    ("SUPERVISED_MPC", "Supervised MPC",  "rev1", False, "#00ff88"),


    # Demonstration & Expo (public demo / stand routines)
    ("EXPO",           "Expo Mode",       "showcase", True,  "#ffcc00"),
]

MODE_TOOLTIPS = {

    "DISCRETE": "Discrete Speed — Manual driving with 7 instant speed presets on keys 1-7",
    "SUPERVISED_MPC": "Supervised MPC — MPC steers autonomously, you control speed via presets (1-7) and can nudge/override steering with A/D",
    "EXPO": "Expo Mode — Car on stand demo with choreographed motor/servo routines and battery monitor",
}

# Section headers (display-only, not selectable)
SECTION_HEADERS = {
    "rev0": ("━━ Rev0 ━━", "#00ff00"),   # Green
    "rev1": ("━━ Rev1 ━━", "#00aaff"),   # Cyan

    "showcase": ("━━ Demonstration & Expo ━━", "#c9a227"),
}

# Default mode on startup
DEFAULT_MODE = "POLY_LOOKAHEAD"

# Visibility rules: which GUI groups are visible for each mode
# "width_group" = LEGACY track width slider
# "advanced_group" = Poly LookAhead / MPC advanced settings
MODE_VISIBILITY = {
    "POLY_LOOKAHEAD": {"width_group": False, "advanced_group": True,  "pd_group": True,  "mpc_group": False, "speed_extras": True,  "cem_group": False, "lla_group": False, "hmpcc_group": False, "pure_pursuit_group": False, "stanley_group": False},
    "LEGACY":         {"width_group": True,  "advanced_group": False, "pd_group": True,  "mpc_group": False, "speed_extras": True,  "cem_group": False, "lla_group": False, "hmpcc_group": False, "pure_pursuit_group": False, "stanley_group": False},
    "DISCRETE":       {"width_group": False, "advanced_group": False, "pd_group": False, "mpc_group": False, "speed_extras": False, "cem_group": False, "lla_group": False, "hmpcc_group": False, "pure_pursuit_group": False, "stanley_group": False, "speed_group": False},
    "MPC":            {"width_group": False, "advanced_group": True,  "pd_group": False, "mpc_group": True,  "speed_extras": False, "cem_group": False, "lla_group": False, "hmpcc_group": False, "pure_pursuit_group": False, "stanley_group": False},
    "SUPERVISED_MPC": {"width_group": False, "advanced_group": True,  "pd_group": False, "mpc_group": True,  "speed_extras": False, "cem_group": False, "lla_group": False, "hmpcc_group": False, "pure_pursuit_group": False, "stanley_group": False},

    "EXPO":           {"width_group": False, "advanced_group": False, "pd_group": False, "mpc_group": False, "speed_extras": False, "cem_group": False, "lla_group": False, "hmpcc_group": False, "pure_pursuit_group": False, "stanley_group": False, "speed_group": False},
}

# Status label styling per mode
MODE_STATUS_STYLE = {
    "POLY_LOOKAHEAD": ("Active: Poly LookAhead", "color: #0f0; font-size: 10px;"),
    "LEGACY":         ("Active: LEGACY",          "color: #aaa; font-size: 10px;"),
    "DISCRETE":       ("Active: Discrete Speed",  "color: #ffcc00; font-weight: bold; font-size: 10px;"),
    "MPC":            ("Active: MPC",             "color: #0f0; font-size: 10px;"),
    "SUPERVISED_MPC": ("Active: Supervised MPC",  "color: #00ff88; font-weight: bold; font-size: 10px;"),

    "EXPO":           ("Active: Expo Mode",       "color: #ffcc00; font-weight: bold; font-size: 10px;"),
}


def build_dropdown_items():
    """
    Build the ordered list of dropdown items with headers inserted.
    
    Returns:
        items: List of (display_label, internal_name_or_None, enabled, color_hex)
               internal_name is None for section headers
        mode_to_index: Dict mapping internal mode name -> dropdown index
    """
    items = []
    mode_to_index = {}
    current_section = None
    
    for (name, label, section, enabled, color) in MODES:
        # Insert section header if entering a new section
        if section != current_section:
            header_label, header_color = SECTION_HEADERS[section]
            items.append((header_label, None, False, header_color))
            current_section = section
        
        idx = len(items)
        mode_to_index[name] = idx
        items.append((label, name, enabled, color))
    
    return items, mode_to_index


def get_index_to_mode(mode_to_index):
    """Reverse mapping: dropdown index -> internal mode name."""
    return {idx: name for name, idx in mode_to_index.items()}
