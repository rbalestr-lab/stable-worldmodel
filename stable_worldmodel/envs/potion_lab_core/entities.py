"""
Entity classes for Potion Lab environment.

Contains:
- Essence: Base ingredient entities with visual patterns
- Player: The controllable alchemist character
- Tool classes: Enchanter, Refiner, Cauldron, Bottler, TrashCan, Delivery Window
- State machines for all interactive tools
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pymunk

from .constants import ESSENCE_TYPES, PhysicsConfig


# ============================================================================
# Essence State and Visual Patterns
# ============================================================================


@dataclass
class EssenceState:
    """
    Represents the state of an essence (raw, combined, or bottled).

    For combined essences, each component maintains its own enchanted/refined state.
    """

    essence_types: list[int]  # List of essence type IDs (1-8)
    enchanted_per_essence: list[bool]  # Per-essence enchanted state
    refined_per_essence: list[bool]  # Per-essence refined state
    is_bottled: bool = False

    def __post_init__(self):
        """Validate that arrays have matching lengths."""
        n = len(self.essence_types)
        assert len(self.enchanted_per_essence) == n
        assert len(self.refined_per_essence) == n

    @property
    def is_combined(self) -> bool:
        return len(self.essence_types) > 1

    def copy(self) -> "EssenceState":
        """Create a deep copy of this essence state."""
        return EssenceState(
            essence_types=self.essence_types.copy(),
            enchanted_per_essence=self.enchanted_per_essence.copy(),
            refined_per_essence=self.refined_per_essence.copy(),
            is_bottled=self.is_bottled,
        )

    def apply_enchantment_to_all(self):
        """Apply enchantment to all essence components."""
        self.enchanted_per_essence = [True] * len(self.essence_types)

    def apply_refinement_to_all(self):
        """Apply refinement to all essence components."""
        self.refined_per_essence = [True] * len(self.essence_types)


# ============================================================================
# Essence Entity
# ============================================================================


class Essence:
    """
    Physical essence object in the world.

    Can be:
    - Raw: Single essence type
    - Combined: Multiple essence types merged together
    - Bottled: Wrapped in a bottle container
    """

    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        essence_state: EssenceState,
        tile_size: float = 32.0,
        mass: float = 0.5,
        friction: float = 0.5,
        elasticity: float = 0.0,
        radius_scale: float = 0.35,
        drag_coefficient: float = 3.0,
    ):
        self.space = space
        self.state = essence_state
        self.tile_size = tile_size
        self.drag_coefficient = drag_coefficient

        # Physics properties
        self.radius = radius_scale * tile_size
        moment = pymunk.moment_for_circle(mass, 0, self.radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = position

        self.body.velocity_func = self._apply_drag

        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.friction = friction
        self.shape.elasticity = elasticity
        self.shape.collision_type = 2  # LAYER_ESSENCE

        self.shape.essence_obj = self

        space.add(self.body, self.shape)

    def _apply_drag(self, body, gravity, damping, dt):
        """Apply custom drag to essence for faster slowdown."""
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        body.velocity = body.velocity * (1.0 - self.drag_coefficient * dt)

    def remove_from_world(self):
        """Remove this essence from the physics simulation."""
        if self.body.space is not None:
            self.space.remove(self.body, self.shape)

    def get_primary_color(self) -> tuple[int, int, int]:
        """Get the color of the first essence type."""
        return ESSENCE_TYPES[self.state.essence_types[0]][1]

    def matches_requirement(self, requirement: dict) -> bool:
        """
        Check if this essence matches a requirement specification.

        Args:
            requirement: Dict with keys:
                - base_essences: List[int]
                - enchanted: List[bool]
                - refined: List[bool]
                - bottled: bool
        """
        # Check if bottled state matches
        if self.state.is_bottled != requirement["bottled"]:
            return False

        # Check if base essences match (order-independent)
        if set(self.state.essence_types) != set(requirement["base_essences"]):
            return False

        # For combined essences, match per-essence patterns
        # Build mapping from requirement essence ID to item essence index
        requirement_map = {}
        for req_essence_id in requirement["base_essences"]:
            try:
                item_idx = self.state.essence_types.index(req_essence_id)
                requirement_map[req_essence_id] = item_idx
            except ValueError:
                return False

        # Check enchanted status for each essence
        for i, req_essence_id in enumerate(requirement["base_essences"]):
            item_idx = requirement_map[req_essence_id]
            if self.state.enchanted_per_essence[item_idx] != requirement["enchanted"][i]:
                return False

        # Check refined status for each essence
        for i, req_essence_id in enumerate(requirement["base_essences"]):
            item_idx = requirement_map[req_essence_id]
            if self.state.refined_per_essence[item_idx] != requirement["refined"][i]:
                return False

        return True


# ============================================================================
# Player Entity
# ============================================================================


class Player:
    """
    The player character (alchemist's apprentice).
    Controlled via continuous velocity inputs.
    """

    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        tile_size: float = 32.0,
        render_size: float = 12.0,
        hitbox_size: float = 12.0,
        mass: float = 1.0,
        friction: float = 0.3,
        elasticity: float = 0.0,
        max_velocity: float = 96.0,
        force_scale: float = 50.0,
        color: tuple[int, int, int] = (65, 105, 225),
        distance_slowdown_scale: float = 20.0,
    ):
        self.space = space
        self.tile_size = tile_size
        self.color = color
        # Separate render size and hitbox size so visuals and collisions can differ
        self.render_size = render_size
        self.hitbox_size = hitbox_size
        # Backwards-compatible alias for existing usages (hitbox size)
        self.size = hitbox_size
        self.max_velocity = max_velocity
        self.force_scale = force_scale
        self.distance_slowdown_scale = distance_slowdown_scale

        # Physics properties
        moment = float("inf")  # No rotation
        self.body = pymunk.Body(mass, moment)
        self.body.position = position

        half_size = hitbox_size / 2
        vertices = [
            (-half_size, -half_size),
            (half_size, -half_size),
            (half_size, half_size),
            (-half_size, half_size),
        ]
        self.shape = pymunk.Poly(self.body, vertices)
        self.shape.friction = friction
        self.shape.elasticity = elasticity
        self.shape.collision_type = 1  # LAYER_PLAYER

        self.shape.player_obj = self

        space.add(self.body, self.shape)

    def apply_action(self, action: np.ndarray):
        """
        Apply force to move player toward cursor position.

        Uses forces to respect physics collisions.

        Args:
            action: [x, y] cursor position in world coordinates
        """
        # Calculate direction from player to cursor
        cursor_pos = np.array(action, dtype=np.float32)
        player_pos = np.array([self.body.position.x, self.body.position.y], dtype=np.float32)

        direction = cursor_pos - player_pos
        distance = np.linalg.norm(direction)

        if distance > 0:
            # Normalize direction
            direction = direction / distance

            # Calculate target velocity proportional to distance for smooth deceleration
            speed_factor = min(1.0, distance / self.distance_slowdown_scale)
            target_velocity = direction * self.max_velocity * speed_factor
        else:
            target_velocity = np.array([0.0, 0.0])

        current_velocity = self.body.velocity
        velocity_diff = (target_velocity[0] - current_velocity.x, target_velocity[1] - current_velocity.y)

        # Apply force based on velocity difference and force_scale
        force = (
            self.body.mass * self.force_scale * velocity_diff[0],
            self.body.mass * self.force_scale * velocity_diff[1],
        )

        self.body.apply_force_at_local_point(force, (0, 0))


# ============================================================================
# Tool State Machines
# ============================================================================


class ToolState(Enum):
    """State enum for single-input tools (Enchanter, Refiner, Bottler)."""

    EMPTY = 0
    PROCESSING = 1
    DONE = 2


class CauldronState(Enum):
    """State enum for the Cauldron."""

    EMPTY = 0
    FILLING = 1
    READY_TO_STIR = 2
    STIRRING = 3
    DONE = 4


class DeliveryState(Enum):
    """State enum for the Delivery Window."""

    WAITING = 0
    CHECKING = 1
    ACCEPTED = 2
    REJECTED = 3


# ============================================================================
# Tool Base Class
# ============================================================================


class Tool:
    """Base class for all tools."""

    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        size: tuple[float, float],
        tile_size: float = 32.0,
        color: tuple[int, int, int] = (150, 150, 150),
        is_sensor: bool = False,
    ):
        self.space = space
        self.position = position
        self.size = size
        self.tile_size = tile_size
        self.color = color

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        self.shape = pymunk.Poly.create_box(self.body, size, radius=0.05 * tile_size)
        self.shape.sensor = is_sensor
        self.shape.collision_type = 3  # LAYER_TOOL
        self.shape.friction = 0.5
        self.shape.elasticity = 0.0

        # Store reference
        self.shape.tool_obj = self

        space.add(self.body, self.shape)

    def update(self, dt: float):
        """Update tool state. Override in subclasses."""
        pass

    def get_display_name(self) -> str:
        """Get the display name for this tool."""
        return self.__class__.__name__

    def reset(self):
        """Reset the tool to its initial state. Override in subclasses if needed."""
        if hasattr(self, "state"):
            self.state = type(self.state)(0)  # Reset to first enum value
        if hasattr(self, "timer"):
            self.timer = 0

    def enable(self):
        """Enable this tool by adding it to the physics space."""
        if self.body not in self.space.bodies:
            self.space.add(self.body, self.shape)

    def disable(self):
        """Disable this tool by removing it from the physics space."""
        if self.body in self.space.bodies:
            self.space.remove(self.body, self.shape)


# ============================================================================
# Enchanter Tool
# ============================================================================


class Enchanter(Tool):
    """
    Enchanter: Adds diagonal stripe pattern to essences.
    Processing time: 120 steps
    """

    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        tile_size: float = 32.0,
        size_multiplier: float = 1.2,
        color: tuple[int, int, int] = (138, 43, 226),
        processing_time: int = 120,
        eject_delay: int = 6,
    ):
        size = (size_multiplier * tile_size, size_multiplier * tile_size)
        super().__init__(space, position, size, tile_size, color, is_sensor=False)

        self.state = ToolState.EMPTY
        self.timer = 0
        self.processing_time = processing_time
        self.eject_delay = eject_delay
        self.current_essence: EssenceState | None = None

    def accept_essence(self, essence: Essence) -> bool:
        """
        Try to accept an essence for processing.

        Returns True if accepted, False if rejected.
        """
        if self.state != ToolState.EMPTY:
            return False

        if essence.state.is_bottled:
            return False

        if all(essence.state.enchanted_per_essence):
            return False

        self.current_essence = essence.state.copy()
        self.state = ToolState.PROCESSING
        self.timer = self.processing_time

        essence.remove_from_world()
        return True

    def update(self, dt: float):
        """Update enchanter state machine."""
        if self.state == ToolState.PROCESSING:
            self.timer -= 1
            if self.timer <= 0:
                # Apply enchantment to all essence components
                self.current_essence.apply_enchantment_to_all()
                self.state = ToolState.DONE
                self.timer = self.eject_delay

        elif self.state == ToolState.DONE:
            self.timer -= 1
            if self.timer <= 0:
                # Eject will be handled by environment
                pass

    def eject_essence(self) -> EssenceState | None:
        """Eject the processed essence and reset state."""
        if self.state != ToolState.DONE or self.timer > 0:
            return None

        essence_state = self.current_essence
        self.current_essence = None
        self.state = ToolState.EMPTY
        return essence_state

    def reset(self):
        """Reset enchanter to initial state."""
        self.state = ToolState.EMPTY
        self.timer = 0
        self.current_essence = None


# ============================================================================
# Refiner Tool
# ============================================================================


class Refiner(Tool):
    """
    Refiner: Adds dotted/stippled pattern to essences.
    Processing time: 150 steps
    """

    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        tile_size: float = 32.0,
        size_multiplier: float = 1.2,
        color: tuple[int, int, int] = (184, 134, 11),
        processing_time: int = 150,
        eject_delay: int = 6,
    ):
        size = (size_multiplier * tile_size, size_multiplier * tile_size)
        super().__init__(space, position, size, tile_size, color, is_sensor=False)

        self.state = ToolState.EMPTY
        self.timer = 0
        self.processing_time = processing_time
        self.eject_delay = eject_delay
        self.current_essence: EssenceState | None = None

    def accept_essence(self, essence: Essence) -> bool:
        """Try to accept an essence for processing."""
        if self.state != ToolState.EMPTY:
            return False

        if essence.state.is_bottled:
            return False

        if all(essence.state.refined_per_essence):
            return False

        self.current_essence = essence.state.copy()
        self.state = ToolState.PROCESSING
        self.timer = self.processing_time

        essence.remove_from_world()
        return True

    def update(self, dt: float):
        """Update refiner state machine."""
        if self.state == ToolState.PROCESSING:
            self.timer -= 1
            if self.timer <= 0:
                # Apply refinement to all essence components
                self.current_essence.apply_refinement_to_all()
                self.state = ToolState.DONE
                self.timer = self.eject_delay

        elif self.state == ToolState.DONE:
            self.timer -= 1
            if self.timer <= 0:
                pass  # Eject handled by environment

    def eject_essence(self) -> EssenceState | None:
        """Eject the processed essence and reset state."""
        if self.state != ToolState.DONE or self.timer > 0:
            return None

        essence_state = self.current_essence
        self.current_essence = None
        self.state = ToolState.EMPTY
        return essence_state

    def reset(self):
        """Reset refiner to initial state."""
        self.state = ToolState.EMPTY
        self.timer = 0
        self.current_essence = None


# ============================================================================
# Cauldron Tool
# ============================================================================


class Cauldron(Tool):
    """
    Cauldron: Combines multiple essences (2-4) into a single multi-colored essence.
    Player must stir by colliding with it for the full mixing time.
    """

    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        tile_size: float = 32.0,
        size_multiplier: float = 1.5,
        color: tuple[int, int, int] = (47, 79, 79),
        stir_time: int = 60,
        eject_delay: int = 6,
    ):
        size = (size_multiplier * tile_size, size_multiplier * tile_size)
        super().__init__(space, position, size, tile_size, color, is_sensor=False)

        self.shape.collision_type = PhysicsConfig.LAYER_CAULDRON

        self.state = CauldronState.EMPTY
        self.timer = 0
        self.stir_time = stir_time
        self.eject_delay = eject_delay
        self.max_essences = 4

        self.essence_slots: list[EssenceState | None] = [None, None, None, None]
        self.stir_progress = 0

    def accept_essence(self, essence: Essence) -> bool:
        """Try to accept an essence into the cauldron."""
        if self.state in (CauldronState.STIRRING, CauldronState.DONE):
            return False

        if essence.state.is_bottled:
            return False

        # Find first empty slot
        empty_slot = None
        for i, slot in enumerate(self.essence_slots):
            if slot is None:
                empty_slot = i
                break

        if empty_slot is None:
            return False  # All slots full

        # Accept the essence into the slot
        self.essence_slots[empty_slot] = essence.state.copy()
        essence.remove_from_world()

        # Update state
        if self.state == CauldronState.EMPTY:
            self.state = CauldronState.FILLING

        # Check if we have at least 2 essences to make it ready to stir
        num_essences = sum(1 for slot in self.essence_slots if slot is not None)
        if num_essences >= 2:
            self.state = CauldronState.READY_TO_STIR

        return True

    def start_stirring(self):
        """Called when player starts colliding with cauldron."""
        if self.state == CauldronState.READY_TO_STIR:
            self.state = CauldronState.STIRRING
            self.stir_progress = 0

    def stir(self):
        """Called each frame player is colliding with cauldron."""
        if self.state == CauldronState.STIRRING:
            self.stir_progress += 1
            if self.stir_progress >= self.stir_time:
                self._combine_essences()
                self.state = CauldronState.DONE
                self.timer = self.eject_delay

    def stop_stirring(self):
        """Called when player stops colliding with cauldron."""
        if self.state == CauldronState.STIRRING:
            self.state = CauldronState.READY_TO_STIR
            self.stir_progress = 0

    def update(self, dt: float):
        """Update cauldron state machine."""
        if self.state == CauldronState.DONE:
            self.timer -= 1
            if self.timer <= 0:
                pass  # Eject handled by environment

    def _combine_essences(self):
        """Combine all essences in the slots into a single combined essence."""
        # Flatten all essence types and their states from slots
        combined_types = []
        combined_enchanted = []
        combined_refined = []

        for essence_state in self.essence_slots:
            if essence_state is not None:
                combined_types.extend(essence_state.essence_types)
                combined_enchanted.extend(essence_state.enchanted_per_essence)
                combined_refined.extend(essence_state.refined_per_essence)

        if len(combined_types) == 0:
            return

        # Sort by essence type so combined essence order is deterministic
        combined = list(zip(combined_types, combined_enchanted, combined_refined))
        combined.sort(key=lambda x: x[0])
        combined_types = [c[0] for c in combined]
        combined_enchanted = [c[1] for c in combined]
        combined_refined = [c[2] for c in combined]

        # Create combined essence state
        self.combined_essence = EssenceState(
            essence_types=combined_types,
            enchanted_per_essence=combined_enchanted,
            refined_per_essence=combined_refined,
            is_bottled=False,
        )

        self.essence_slots = [None, None, None, None]

    def eject_essence(self) -> EssenceState | None:
        """Eject the combined essence and reset state."""
        if self.state != CauldronState.DONE or self.timer > 0:
            return None

        essence_state = self.combined_essence
        self.combined_essence = None
        self.state = CauldronState.EMPTY
        self.stir_progress = 0
        return essence_state

    def get_num_essences(self) -> int:
        """Get the number of essences currently in the cauldron."""
        return sum(1 for slot in self.essence_slots if slot is not None)

    def reset(self):
        """Reset cauldron to initial state."""
        self.state = CauldronState.EMPTY
        self.timer = 0
        self.essence_slots = [None, None, None, None]
        self.stir_progress = 0
        self.combined_essence = None


# ============================================================================
# Bottler Tool
# ============================================================================


class Bottler(Tool):
    """
    Bottler: Places any essence into a potion bottle.
    Bottling time: 90 steps
    """

    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        tile_size: float = 32.0,
        size_multiplier: float = 1.0,
        color: tuple[int, int, int] = (176, 196, 222),
        bottling_time: int = 90,
        eject_delay: int = 6,
    ):
        size = (size_multiplier * tile_size, size_multiplier * tile_size)
        super().__init__(space, position, size, tile_size, color, is_sensor=False)

        self.state = ToolState.EMPTY
        self.timer = 0
        self.bottling_time = bottling_time
        self.eject_delay = eject_delay
        self.current_essence: EssenceState | None = None

    def accept_essence(self, essence: Essence) -> bool:
        """Try to accept an essence for bottling."""
        if self.state != ToolState.EMPTY:
            return False

        if essence.state.is_bottled:
            return False

        self.current_essence = essence.state.copy()
        self.state = ToolState.PROCESSING
        self.timer = self.bottling_time

        essence.remove_from_world()
        return True

    def update(self, dt: float):
        """Update bottler state machine."""
        if self.state == ToolState.PROCESSING:
            self.timer -= 1
            if self.timer <= 0:
                self.current_essence.is_bottled = True
                self.state = ToolState.DONE
                self.timer = self.eject_delay

        elif self.state == ToolState.DONE:
            self.timer -= 1
            if self.timer <= 0:
                pass  # Eject handled by environment

    def eject_essence(self) -> EssenceState | None:
        """Eject the bottled essence and reset state."""
        if self.state != ToolState.DONE or self.timer > 0:
            return None

        essence_state = self.current_essence
        self.current_essence = None
        self.state = ToolState.EMPTY
        return essence_state

    def reset(self):
        """Reset bottler to initial state."""
        self.state = ToolState.EMPTY
        self.timer = 0
        self.current_essence = None


# ============================================================================
# Trash Can
# ============================================================================


class TrashCan(Tool):
    """
    Trash Can: Destroys any essence or bottle permanently.
    Instant destruction (0 steps).
    """

    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        tile_size: float = 32.0,
        size_multiplier: float = 0.8,
        color: tuple[int, int, int] = (105, 105, 105),
    ):
        size = (size_multiplier * tile_size, size_multiplier * tile_size)
        super().__init__(space, position, size, tile_size, color, is_sensor=False)

    def accept_essence(self, essence: Essence) -> bool:
        """Accept and immediately destroy an essence."""
        essence.remove_from_world()
        return True

    def update(self, dt: float):
        """Trash can has no state to update."""
        pass


# ============================================================================
# Dispenser
# ============================================================================


class Dispenser:
    """
    Dispenser: Spawns essences of a specific type when player collides with it.
    Cooldown: 30 steps between dispenses.
    """

    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        essence_type: int,
        tile_size: float = 32.0,
        size_multiplier: float = 0.8,
        cooldown_duration: int = 30,
    ):
        self.space = space
        self.position = position
        self.essence_type = essence_type
        self.tile_size = tile_size
        self.cooldown = 0
        self.cooldown_duration = cooldown_duration

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        size = (size_multiplier * tile_size, size_multiplier * tile_size)
        self.size = size
        self.shape = pymunk.Poly.create_box(self.body, size)
        self.shape.sensor = False
        self.shape.collision_type = PhysicsConfig.LAYER_DISPENSER
        self.shape.friction = 0.5
        self.shape.elasticity = 0.0

        self.shape.dispenser_obj = self

        self.color = ESSENCE_TYPES[essence_type][1]

        space.add(self.body, self.shape)

    def update(self, dt: float):
        """Update cooldown timer."""
        if self.cooldown > 0:
            self.cooldown -= 1

    def can_dispense(self) -> bool:
        """Check if dispenser is ready to dispense."""
        return self.cooldown <= 0

    def dispense(self) -> EssenceState | None:
        """
        Dispense an essence if ready.

        Returns the essence state to spawn, or None if on cooldown.
        """
        if not self.can_dispense():
            return None

        # Create raw essence state
        essence_state = EssenceState(
            essence_types=[self.essence_type],
            enchanted_per_essence=[False],
            refined_per_essence=[False],
            is_bottled=False,
        )

        self.cooldown = self.cooldown_duration

        return essence_state

    def enable(self):
        """Enable this dispenser by adding it to the physics space."""
        if self.body not in self.space.bodies:
            self.space.add(self.body, self.shape)

    def disable(self):
        """Disable this dispenser by removing it from the physics space."""
        if self.body in self.space.bodies:
            self.space.remove(self.body, self.shape)


# ============================================================================
# Delivery Window
# ============================================================================


class DeliveryWindow(Tool):
    """
    Delivery Window: Validates delivered items against requirements.
    """

    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        tile_size: float = 32.0,
        width_multiplier: float = 1.5,
        height_multiplier: float = 1.0,
        color: tuple[int, int, int] = (60, 179, 113),
        feedback_duration: int = 12,
    ):
        size = (width_multiplier * tile_size, height_multiplier * tile_size)
        super().__init__(space, position, size, tile_size, color, is_sensor=False)

        self.state = DeliveryState.WAITING
        self.timer = 0
        self.feedback_duration = feedback_duration

        self.required_items: list[dict] = []

    def set_requirements(self, requirements: list[dict]):
        """Set the list of required items for this round."""
        self.required_items = requirements

    def validate_delivery(self, essence: Essence) -> bool:
        """
        Validate if the delivered essence matches any requirement.

        Returns True if valid (accepted), False if invalid (trashed).
        """
        if self.state != DeliveryState.WAITING:
            return False

        self.state = DeliveryState.CHECKING

        for requirement in self.required_items:
            if requirement.get("completed", False):
                continue

            if essence.matches_requirement(requirement):
                # Valid delivery
                requirement["completed"] = True
                essence.remove_from_world()
                self.state = DeliveryState.ACCEPTED
                self.timer = self.feedback_duration
                return True

        # Invalid delivery - trash it
        self.state = DeliveryState.REJECTED
        self.timer = self.feedback_duration
        essence.remove_from_world()  # Trash invalid items
        return True  # Return True so it gets removed from essences list

    def update(self, dt: float):
        """Update delivery window state machine."""
        if self.state in (DeliveryState.ACCEPTED, DeliveryState.REJECTED):
            self.timer -= 1
            if self.timer <= 0:
                self.state = DeliveryState.WAITING

    def all_requirements_met(self) -> bool:
        """Check if all requirements have been completed."""
        return all(req.get("completed", False) for req in self.required_items)
