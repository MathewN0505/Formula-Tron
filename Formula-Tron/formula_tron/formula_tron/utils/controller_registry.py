"""
Controller Registry — maps mode name strings to controller instances.

Central lookup for all registered control modes. Used by vision_controller
to dispatch to the correct controller without if/elif chains.
"""

from typing import Dict, List, Optional
from .base_controller import BaseController


class ControllerRegistry:
    """
    Registry of available control modes.
    
    Usage:
        registry = ControllerRegistry()
        registry.register(PDController("POLY_LOOKAHEAD"))
        registry.register(PDController("LEGACY", detection_mode="LEGACY"))
        registry.register(MPCController(...))
        
        controller = registry.get("MPC")
        output = controller.compute(state, dt)
    """

    def __init__(self):
        self._controllers: Dict[str, BaseController] = {}

    def register(self, controller: BaseController):
        """Register a controller. Raises if name already registered."""
        name = controller.name
        if name in self._controllers:
            raise ValueError(f"Controller '{name}' already registered")
        self._controllers[name] = controller

    def get(self, mode_name: str) -> BaseController:
        """Get controller by mode name. Raises KeyError if not found."""
        if mode_name not in self._controllers:
            available = ', '.join(self._controllers.keys())
            raise KeyError(f"Unknown control mode '{mode_name}'. Available: {available}")
        return self._controllers[mode_name]

    def has(self, mode_name: str) -> bool:
        """Check if a mode is registered."""
        return mode_name in self._controllers

    def available_modes(self) -> List[str]:
        """List all registered mode names."""
        return list(self._controllers.keys())

    def update_all_params(self, params: dict):
        """Broadcast parameter update to all controllers."""
        for controller in self._controllers.values():
            controller.update_params(params)

    def reset_all(self):
        """Reset all controllers (e.g. on autonomous start)."""
        for controller in self._controllers.values():
            controller.reset()
