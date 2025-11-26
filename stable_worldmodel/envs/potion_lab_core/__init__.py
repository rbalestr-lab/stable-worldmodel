"""
Potion Lab environment package.

Contains the core game logic and entities for the Potion Brewing Laboratory.
"""

from .constants import ESSENCE_TYPES, PhysicsConfig
from .entities import (
    Bottler,
    Cauldron,
    DeliveryWindow,
    Dispenser,
    Enchanter,
    Essence,
    EssenceState,
    Player,
    Refiner,
    TrashCan,
)
from .game_logic import (
    CollisionHandler,
    RoundManager,
    add_walls,
    create_default_layout,
    draw_dispenser,
    draw_essence,
    draw_player,
    draw_tool,
    draw_ui,
    setup_physics_space,
)


__all__ = [
    "Essence",
    "EssenceState",
    "Player",
    "Enchanter",
    "Refiner",
    "Cauldron",
    "Bottler",
    "TrashCan",
    "Dispenser",
    "DeliveryWindow",
    "ESSENCE_TYPES",
    "PhysicsConfig",
    "setup_physics_space",
    "add_walls",
    "CollisionHandler",
    "RoundManager",
    "create_default_layout",
    "draw_essence",
    "draw_tool",
    "draw_player",
    "draw_dispenser",
    "draw_ui",
]
