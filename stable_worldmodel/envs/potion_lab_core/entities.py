"""
Entity classes for Potion Lab environment.

Contains:
- Essence: Base ingredient entities with visual patterns
- Player: The controllable alchemist character
- Tool classes: Enchanter, Refiner, Cauldron, Bottler, TrashCan, Delivery Window
- State machines for all interactive tools
"""

from enum import Enum
from typing import List, Tuple, Optional
import numpy as np
import pymunk
import pygame
from dataclasses import dataclass, field


# ============================================================================
# Essence State and Visual Patterns
# ============================================================================

@dataclass
class EssenceState:
    """
    Represents the state of an essence (raw, combined, or bottled).
    
    For combined essences, each component maintains its own enchanted/refined state.
    """
    essence_types: List[int]  # List of essence type IDs (1-8)
    enchanted_per_essence: List[bool]  # Per-essence enchanted state
    refined_per_essence: List[bool]  # Per-essence refined state
    is_bottled: bool = False
    
    def __post_init__(self):
        """Validate that arrays have matching lengths."""
        n = len(self.essence_types)
        assert len(self.enchanted_per_essence) == n
        assert len(self.refined_per_essence) == n
    
    @property
    def is_combined(self) -> bool:
        return len(self.essence_types) > 1
    
    def copy(self) -> 'EssenceState':
        """Create a deep copy of this essence state."""
        return EssenceState(
            essence_types=self.essence_types.copy(),
            enchanted_per_essence=self.enchanted_per_essence.copy(),
            refined_per_essence=self.refined_per_essence.copy(),
            is_bottled=self.is_bottled
        )
    
    def apply_enchantment_to_all(self):
        """Apply enchantment to all essence components."""
        self.enchanted_per_essence = [True] * len(self.essence_types)
    
    def apply_refinement_to_all(self):
        """Apply refinement to all essence components."""
        self.refined_per_essence = [True] * len(self.essence_types)


# Essence type definitions (ID -> (Name, Color))
ESSENCE_TYPES = {
    1: ("Fire", (231, 76, 60)),      # Red
    2: ("Water", (52, 152, 219)),    # Blue
    3: ("Earth", (46, 204, 113)),    # Green
    4: ("Air", (241, 196, 15)),      # Yellow
    5: ("Light", (236, 240, 241)),   # White
    6: ("Shadow", (52, 73, 94)),     # Dark Gray
    7: ("Arcane", (155, 89, 182)),   # Purple
    8: ("Nature", (22, 160, 133)),   # Teal
}


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
        position: Tuple[float, float],
        essence_state: EssenceState,
        tile_size: float = 32.0
    ):
        self.space = space
        self.state = essence_state
        self.tile_size = tile_size
        
        # Physics properties
        self.radius = 0.15 * tile_size  # 0.15 tiles
        mass = 0.5
        moment = pymunk.moment_for_circle(mass, 0, self.radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = position
        
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.friction = 0.5  # Higher friction to stop sliding
        self.shape.elasticity = 0.0
        self.shape.collision_type = 2  # LAYER_ESSENCE
        
        # Add damping to the body to prevent sliding
        self.body.velocity_func = self._velocity_update_func
        
        # Store reference to this essence in the shape
        self.shape.essence_obj = self
        
        space.add(self.body, self.shape)
    
    def _velocity_update_func(self, body, gravity, damping, dt):
        """Custom velocity update to add strong damping."""
        # Apply default physics
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        
        # Apply additional velocity damping to stop sliding quickly
        velocity_damping = 0.85
        body.velocity = body.velocity * velocity_damping
        
        # Stop completely if moving very slowly
        if body.velocity.length < 1.0:
            body.velocity = (0, 0)
    
    def remove_from_world(self):
        """Remove this essence from the physics simulation."""
        if self.body.space is not None:
            self.space.remove(self.body, self.shape)
    
    def get_primary_color(self) -> Tuple[int, int, int]:
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
        if self.state.is_bottled != requirement['bottled']:
            return False
        
        # Check if base essences match (order-independent)
        if set(self.state.essence_types) != set(requirement['base_essences']):
            return False
        
        # For combined essences, match per-essence patterns
        # Build mapping from requirement essence ID to item essence index
        requirement_map = {}
        for req_essence_id in requirement['base_essences']:
            try:
                item_idx = self.state.essence_types.index(req_essence_id)
                requirement_map[req_essence_id] = item_idx
            except ValueError:
                return False
        
        # Check enchanted status for each essence
        for i, req_essence_id in enumerate(requirement['base_essences']):
            item_idx = requirement_map[req_essence_id]
            if self.state.enchanted_per_essence[item_idx] != requirement['enchanted'][i]:
                return False
        
        # Check refined status for each essence
        for i, req_essence_id in enumerate(requirement['base_essences']):
            item_idx = requirement_map[req_essence_id]
            if self.state.refined_per_essence[item_idx] != requirement['refined'][i]:
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
        position: Tuple[float, float],
        tile_size: float = 32.0,
        color: Tuple[int, int, int] = (65, 105, 225)  # Royal Blue
    ):
        self.space = space
        self.tile_size = tile_size
        self.color = color
        
        # Physics properties
        self.radius = 0.35 * tile_size  # 0.35 tiles
        mass = 1.0
        moment = float('inf')  # No rotation
        self.body = pymunk.Body(mass, moment)
        self.body.position = position
        
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.friction = 0.3
        self.shape.elasticity = 0.0
        self.shape.collision_type = 1  # LAYER_PLAYER
        
        # Store reference
        self.shape.player_obj = self
        
        # Movement properties
        self.max_velocity = 3.0 * tile_size  # tiles/second
        
        space.add(self.body, self.shape)
    
    def apply_action(self, action: np.ndarray):
        """
        Apply velocity-based action to the player.
        
        Args:
            action: [vx, vy] normalized to [-1, 1]
        """
        # Clamp action
        action = np.clip(action, -1.0, 1.0)
        
        # Convert to velocity
        velocity = action * self.max_velocity
        self.body.velocity = (float(velocity[0]), float(velocity[1]))


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
    MIXING = 2
    DONE = 3


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
        position: Tuple[float, float],
        size: Tuple[float, float],
        tile_size: float = 32.0,
        color: Tuple[int, int, int] = (150, 150, 150),
        is_sensor: bool = False
    ):
        self.space = space
        self.position = position
        self.size = size
        self.tile_size = tile_size
        self.color = color
        
        # Create static body for tool
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        self.shape = pymunk.Poly.create_box(self.body, size, radius=0.05 * tile_size)
        self.shape.sensor = is_sensor  # Most tools are solid, some are sensors
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


# ============================================================================
# Enchanter Tool
# ============================================================================

class Enchanter(Tool):
    """
    Enchanter: Adds diagonal stripe pattern to essences.
    Processing time: 120 steps
    """
    
    def __init__(self, space: pymunk.Space, position: Tuple[float, float], tile_size: float = 32.0):
        size = (1.2 * tile_size, 1.2 * tile_size)
        color = (138, 43, 226)  # Blue Violet
        super().__init__(space, position, size, tile_size, color, is_sensor=False)
        
        self.state = ToolState.EMPTY
        self.timer = 0
        self.processing_time = 120  # steps
        self.eject_delay = 6  # steps
        self.current_essence: Optional[EssenceState] = None
    
    def accept_essence(self, essence: Essence) -> bool:
        """
        Try to accept an essence for processing.
        
        Returns True if accepted, False if rejected.
        """
        if self.state != ToolState.EMPTY:
            return False
        
        # Accept the essence
        self.current_essence = essence.state.copy()
        self.state = ToolState.PROCESSING
        self.timer = self.processing_time
        
        # Remove the physical essence from the world
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
    
    def eject_essence(self) -> Optional[EssenceState]:
        """Eject the processed essence and reset state."""
        if self.state != ToolState.DONE or self.timer > 0:
            return None
        
        essence_state = self.current_essence
        self.current_essence = None
        self.state = ToolState.EMPTY
        return essence_state


# ============================================================================
# Refiner Tool
# ============================================================================

class Refiner(Tool):
    """
    Refiner: Adds dotted/stippled pattern to essences.
    Processing time: 150 steps
    """
    
    def __init__(self, space: pymunk.Space, position: Tuple[float, float], tile_size: float = 32.0):
        size = (1.2 * tile_size, 1.2 * tile_size)
        color = (184, 134, 11)  # Dark Goldenrod
        super().__init__(space, position, size, tile_size, color, is_sensor=False)
        
        self.state = ToolState.EMPTY
        self.timer = 0
        self.processing_time = 150  # steps
        self.eject_delay = 6  # steps
        self.current_essence: Optional[EssenceState] = None
    
    def accept_essence(self, essence: Essence) -> bool:
        """Try to accept an essence for processing."""
        if self.state != ToolState.EMPTY:
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
    
    def eject_essence(self) -> Optional[EssenceState]:
        """Eject the processed essence and reset state."""
        if self.state != ToolState.DONE or self.timer > 0:
            return None
        
        essence_state = self.current_essence
        self.current_essence = None
        self.state = ToolState.EMPTY
        return essence_state


# ============================================================================
# Cauldron Tool
# ============================================================================

class Cauldron(Tool):
    """
    Cauldron: Combines multiple essences (2-4) into a single multi-colored essence.
    Staging time: 60 steps (resets on each new essence)
    Mixing time: 60 steps
    """
    
    def __init__(self, space: pymunk.Space, position: Tuple[float, float], tile_size: float = 32.0):
        size = (1.5 * tile_size, 1.5 * tile_size)
        color = (47, 79, 79)  # Dark Slate Gray
        super().__init__(space, position, size, tile_size, color, is_sensor=False)
        
        self.state = CauldronState.EMPTY
        self.timer = 0
        self.staging_time = 60  # steps
        self.mixing_time = 60  # steps
        self.eject_delay = 6  # steps
        self.max_essences = 4
        
        self.essence_list: List[EssenceState] = []
    
    def accept_essence(self, essence: Essence) -> bool:
        """Try to accept an essence into the cauldron."""
        if self.state == CauldronState.DONE:
            return False
        
        # Check capacity
        if len(self.essence_list) >= self.max_essences:
            return False  # Reject, cauldron full
        
        # Accept the essence
        self.essence_list.append(essence.state.copy())
        essence.remove_from_world()
        
        # Update state
        if self.state == CauldronState.EMPTY:
            self.state = CauldronState.FILLING
        
        # Reset staging timer
        self.timer = self.staging_time
        return True
    
    def update(self, dt: float):
        """Update cauldron state machine."""
        if self.state == CauldronState.FILLING:
            self.timer -= 1
            if self.timer <= 0:
                # Start mixing
                self.state = CauldronState.MIXING
                self.timer = self.mixing_time
        
        elif self.state == CauldronState.MIXING:
            self.timer -= 1
            if self.timer <= 0:
                # Create combined essence
                self._combine_essences()
                self.state = CauldronState.DONE
                self.timer = self.eject_delay
        
        elif self.state == CauldronState.DONE:
            self.timer -= 1
            if self.timer <= 0:
                pass  # Eject handled by environment
    
    def _combine_essences(self):
        """Combine all essences in the list into a single combined essence."""
        if len(self.essence_list) == 0:
            return
        
        # Flatten all essence types and their states
        combined_types = []
        combined_enchanted = []
        combined_refined = []
        
        for essence_state in self.essence_list:
            combined_types.extend(essence_state.essence_types)
            combined_enchanted.extend(essence_state.enchanted_per_essence)
            combined_refined.extend(essence_state.refined_per_essence)
        
        # Create combined essence state
        self.combined_essence = EssenceState(
            essence_types=combined_types,
            enchanted_per_essence=combined_enchanted,
            refined_per_essence=combined_refined,
            is_bottled=False
        )
        
        # Clear the essence list
        self.essence_list = []
    
    def eject_essence(self) -> Optional[EssenceState]:
        """Eject the combined essence and reset state."""
        if self.state != CauldronState.DONE or self.timer > 0:
            return None
        
        essence_state = self.combined_essence
        self.combined_essence = None
        self.state = CauldronState.EMPTY
        return essence_state


# ============================================================================
# Bottler Tool
# ============================================================================

class Bottler(Tool):
    """
    Bottler: Places any essence into a potion bottle.
    Bottling time: 90 steps
    """
    
    def __init__(self, space: pymunk.Space, position: Tuple[float, float], tile_size: float = 32.0):
        size = (1.0 * tile_size, 1.0 * tile_size)
        color = (176, 196, 222)  # Light Steel Blue
        super().__init__(space, position, size, tile_size, color, is_sensor=False)
        
        self.state = ToolState.EMPTY
        self.timer = 0
        self.bottling_time = 90  # steps
        self.eject_delay = 6  # steps
        self.current_essence: Optional[EssenceState] = None
    
    def accept_essence(self, essence: Essence) -> bool:
        """Try to accept an essence for bottling."""
        if self.state != ToolState.EMPTY:
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
                # Bottle the essence
                self.current_essence.is_bottled = True
                self.state = ToolState.DONE
                self.timer = self.eject_delay
        
        elif self.state == ToolState.DONE:
            self.timer -= 1
            if self.timer <= 0:
                pass  # Eject handled by environment
    
    def eject_essence(self) -> Optional[EssenceState]:
        """Eject the bottled essence and reset state."""
        if self.state != ToolState.DONE or self.timer > 0:
            return None
        
        essence_state = self.current_essence
        self.current_essence = None
        self.state = ToolState.EMPTY
        return essence_state


# ============================================================================
# Trash Can
# ============================================================================

class TrashCan(Tool):
    """
    Trash Can: Destroys any essence or bottle permanently.
    Instant destruction (0 steps).
    """
    
    def __init__(self, space: pymunk.Space, position: Tuple[float, float], tile_size: float = 32.0):
        size = (0.8 * tile_size, 0.8 * tile_size)
        color = (105, 105, 105)  # Dim Gray
        super().__init__(space, position, size, tile_size, color, is_sensor=True)
    
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
        position: Tuple[float, float],
        essence_type: int,
        tile_size: float = 32.0
    ):
        self.space = space
        self.position = position
        self.essence_type = essence_type
        self.tile_size = tile_size
        self.cooldown = 0
        self.cooldown_duration = 30  # steps
        
        # Create sensor shape for collision detection
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        size = (0.8 * tile_size, 0.8 * tile_size)
        self.shape = pymunk.Poly.create_box(self.body, size)
        self.shape.sensor = True  # Sensors for triggering dispense
        self.shape.collision_type = 3  # LAYER_TOOL
        
        # Store reference
        self.shape.dispenser_obj = self
        
        # Get color from essence type
        self.color = ESSENCE_TYPES[essence_type][1]
        
        space.add(self.body, self.shape)
    
    def update(self, dt: float):
        """Update cooldown timer."""
        if self.cooldown > 0:
            self.cooldown -= 1
    
    def can_dispense(self) -> bool:
        """Check if dispenser is ready to dispense."""
        return self.cooldown <= 0
    
    def dispense(self) -> Optional[EssenceState]:
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
            is_bottled=False
        )
        
        # Start cooldown
        self.cooldown = self.cooldown_duration
        
        return essence_state


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
        position: Tuple[float, float],
        tile_size: float = 32.0
    ):
        size = (1.5 * tile_size, 1.0 * tile_size)
        color = (60, 179, 113)  # Medium Sea Green
        super().__init__(space, position, size, tile_size, color, is_sensor=True)
        
        self.state = DeliveryState.WAITING
        self.timer = 0
        self.feedback_duration = 12  # steps
        
        # Requirements will be set by the environment
        self.required_items: List[dict] = []
    
    def set_requirements(self, requirements: List[dict]):
        """Set the list of required items for this round."""
        self.required_items = requirements
    
    def validate_delivery(self, essence: Essence) -> bool:
        """
        Validate if the delivered essence matches any requirement.
        
        Returns True if valid, False otherwise.
        """
        if self.state != DeliveryState.WAITING:
            return False
        
        self.state = DeliveryState.CHECKING
        
        # Check against all requirements
        for requirement in self.required_items:
            if requirement.get('completed', False):
                continue
            
            if essence.matches_requirement(requirement):
                # Valid delivery
                requirement['completed'] = True
                essence.remove_from_world()
                self.state = DeliveryState.ACCEPTED
                self.timer = self.feedback_duration
                return True
        
        # Invalid delivery
        self.state = DeliveryState.REJECTED
        self.timer = self.feedback_duration
        # Don't remove essence, let it bounce back
        return False
    
    def update(self, dt: float):
        """Update delivery window state machine."""
        if self.state in (DeliveryState.ACCEPTED, DeliveryState.REJECTED):
            self.timer -= 1
            if self.timer <= 0:
                self.state = DeliveryState.WAITING
    
    def all_requirements_met(self) -> bool:
        """Check if all requirements have been completed."""
        return all(req.get('completed', False) for req in self.required_items)

