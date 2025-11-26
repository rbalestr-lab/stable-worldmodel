"""
Constants and configuration for Potion Lab environment.

Contains:
- Physics configuration (collision layers, physics constants)
- Essence type definitions (colors and names)
"""


# ============================================================================
# Physics Configuration
# ============================================================================


class PhysicsConfig:
    """Configuration for the physics simulation."""

    # Collision layers
    LAYER_PLAYER = 1
    LAYER_ESSENCE = 2
    LAYER_TOOL = 3
    LAYER_WALL = 4
    LAYER_DISPENSER = 5
    LAYER_CAULDRON = 6

    # Physics constants
    GRAVITY = (0, 0)
    DAMPING = 0.8
    ITERATIONS = 20
    TIMESTEP = 1 / 60


# ============================================================================
# Essence Type Definitions
# ============================================================================

# Essence type definitions (ID -> (Name, Color))
ESSENCE_TYPES = {
    1: ("Fire", (231, 76, 60)),  # Red
    2: ("Water", (52, 152, 219)),  # Blue
    3: ("Earth", (46, 204, 113)),  # Green
    4: ("Air", (241, 196, 15)),  # Yellow
    5: ("Light", (236, 240, 241)),  # White
    6: ("Shadow", (52, 73, 94)),  # Dark Gray
    7: ("Arcane", (155, 89, 182)),  # Purple
    8: ("Nature", (22, 160, 133)),  # Teal
}
