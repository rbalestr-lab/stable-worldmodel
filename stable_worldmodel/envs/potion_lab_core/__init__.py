"""
Potion Lab environment package.

Contains the core game logic and entities for the Potion Brewing Laboratory.
"""

from .entities import (
    Essence,
    EssenceState,
    Player,
    Enchanter,
    Refiner,
    Cauldron,
    Bottler,
    TrashCan,
    Dispenser,
    DeliveryWindow,
    ESSENCE_TYPES,
)

from .game_logic import (
    PhysicsConfig,
    setup_physics_space,
    add_walls,
    CollisionHandler,
    RoundManager,
    create_default_layout,
    draw_essence,
    draw_tool,
    draw_player,
    draw_dispenser,
    draw_ui,
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

