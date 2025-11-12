"""
Game logic for Potion Lab environment.

Contains:
- Physics setup and configuration
- Collision handlers for player-tool interactions
- Rendering functions for essences with visual patterns
- Round configuration and management
"""

from typing import List, Tuple, Dict, Optional
import json
import os
import numpy as np
import pygame
import pymunk
from pymunk.vec2d import Vec2d

from .entities import (
    Essence, EssenceState, Player, 
    Enchanter, Refiner, Cauldron, Bottler, TrashCan, Dispenser, DeliveryWindow,
    ESSENCE_TYPES, ToolState, CauldronState
)


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
    
    # Physics constants
    GRAVITY = (0, 0)  # Top-down view, no gravity
    DAMPING = 0.95  # Objects slow down naturally (higher = less damping)
    ITERATIONS = 10  # Collision accuracy
    TIMESTEP = 1/60  # 60 Hz


def setup_physics_space() -> pymunk.Space:
    """Create and configure a pymunk physics space."""
    space = pymunk.Space()
    space.gravity = PhysicsConfig.GRAVITY
    space.damping = PhysicsConfig.DAMPING
    space.iterations = PhysicsConfig.ITERATIONS
    return space


def add_walls(space: pymunk.Space, map_width: float, map_height: float, wall_thickness: float = 10.0):
    """
    Add boundary walls to the physics space.
    
    Args:
        space: Pymunk space
        map_width: Width of the map in pixels
        map_height: Height of the map in pixels
        wall_thickness: Thickness of walls
    """
    walls = [
        # Bottom
        pymunk.Segment(space.static_body, (0, 0), (map_width, 0), wall_thickness),
        # Right
        pymunk.Segment(space.static_body, (map_width, 0), (map_width, map_height), wall_thickness),
        # Top
        pymunk.Segment(space.static_body, (map_width, map_height), (0, map_height), wall_thickness),
        # Left
        pymunk.Segment(space.static_body, (0, map_height), (0, 0), wall_thickness),
    ]
    
    for wall in walls:
        wall.friction = 1.0
        wall.elasticity = 0.0
        wall.collision_type = PhysicsConfig.LAYER_WALL
    
    space.add(*walls)
    return walls


# ============================================================================
# Collision Handlers
# ============================================================================

class CollisionHandler:
    """Manages collision detection and response between game objects."""
    
    def __init__(self, env):
        """
        Initialize collision handler.
        
        Args:
            env: Reference to the PotionLabEnv
        """
        self.env = env
        self.processed_collisions = set()  # Track processed collisions this frame
        self.player_stirring_cauldron = False  # Track if player is currently stirring
    
    def setup_handlers(self, space: pymunk.Space):
        """Set up collision handlers - this is now a no-op as we check manually."""
        # No pymunk handlers needed - we'll check for overlaps manually in update()
        pass
    
    def update(self):
        """Check for collisions manually each frame."""
        self.processed_collisions.clear()
        
        # Check player collisions with dispensers
        self._check_player_dispenser_collisions()
        
        # Check player collisions with cauldron (for stirring)
        self._check_player_cauldron_collisions()
        
        # Check essence collisions with tools
        self._check_essence_tool_collisions()
    
    def _check_player_dispenser_collisions(self):
        """Check if player is touching any dispensers."""
        player = self.env.player
        
        for dispenser in self.env.dispensers:
            # Check if player shape overlaps with dispenser shape
            if self._shapes_overlap(player.shape, dispenser.shape):
                collision_key = ('player', id(dispenser))
                if collision_key not in self.processed_collisions:
                    self.processed_collisions.add(collision_key)
                    
                    essence_state = dispenser.dispense()
                    if essence_state is not None:
                        # Spawn essence at dispenser location
                        position = dispenser.position
                        # Offset slightly so player can push it away
                        offset = Vec2d(0, dispenser.tile_size * 0.5)
                        spawn_pos = (position[0] + offset.x, position[1] + offset.y)
                        
                        essence = Essence(
                            self.env.space,
                            spawn_pos,
                            essence_state,
                            tile_size=self.env.tile_size
                        )
                        self.env.essences.append(essence)
    
    def _check_player_cauldron_collisions(self):
        """Check if player is touching the cauldron to stir it."""
        player = self.env.player
        cauldron = self.env.tools.get('cauldron')
        
        if cauldron is None:
            return
        
        # Check if player shape overlaps with cauldron shape
        if self._shapes_overlap(player.shape, cauldron.shape):
            if not self.player_stirring_cauldron:
                # Just started stirring
                cauldron.start_stirring()
                self.player_stirring_cauldron = True
            else:
                # Continue stirring
                cauldron.stir()
        else:
            if self.player_stirring_cauldron:
                # Stopped stirring
                cauldron.stop_stirring()
                self.player_stirring_cauldron = False
    
    def _check_essence_tool_collisions(self):
        """Check if any essences are touching tools."""
        # Check each essence against each tool
        for essence in self.env.essences[:]:  # Copy list as we may modify it
            # Check against all tools
            for tool_name, tool in self.env.tools.items():
                if self._shapes_overlap(essence.shape, tool.shape):
                    collision_key = (id(essence), id(tool))
                    if collision_key not in self.processed_collisions:
                        self.processed_collisions.add(collision_key)
                        
                        # Handle delivery window specially
                        if tool_name == 'delivery_window':
                            if hasattr(tool, 'validate_delivery'):
                                # Returns True if accepted, False if rejected
                                accepted = tool.validate_delivery(essence)
                                # Remove from list if accepted
                                if accepted and essence in self.env.essences:
                                    self.env.essences.remove(essence)
                                    break  # Essence was consumed, move to next essence
                        # Try to accept the essence for other tools
                        elif hasattr(tool, 'accept_essence'):
                            accepted = tool.accept_essence(essence)
                            if accepted and essence in self.env.essences:
                                self.env.essences.remove(essence)
                                break  # Essence was consumed, move to next essence
    
    def _shapes_overlap(self, shape1, shape2):
        """Check if two shapes overlap using bounding boxes."""
        # Get bounding boxes
        bb1 = shape1.bb
        bb2 = shape2.bb
        
        # Check if bounding boxes overlap
        return not (bb1.right < bb2.left or
                   bb1.left > bb2.right or
                   bb1.top < bb2.bottom or
                   bb1.bottom > bb2.top)


# ============================================================================
# Rendering Functions
# ============================================================================

def draw_essence(
    canvas: pygame.Surface,
    essence: Essence,
    tile_size: float
):
    """
    Draw an essence with its visual patterns.
    
    Visual patterns:
    - Base color: Solid fill
    - Enchanted: Diagonal stripes
    - Refined: Dotted pattern
    - Bottled: Bottle outline
    """
    state = essence.state
    pos = essence.body.position
    radius = essence.radius
    
    # Convert position to pygame coordinates
    # Pymunk and Pygame both use +Y down for top-down view
    screen_pos = (int(pos.x), int(pos.y))
    screen_radius = int(radius)
    
    if state.is_bottled:
        # Draw bottle outline
        bottle_rect = pygame.Rect(
            screen_pos[0] - screen_radius * 1.2,
            screen_pos[1] - screen_radius * 1.5,
            screen_radius * 2.4,
            screen_radius * 3
        )
        pygame.draw.rect(canvas, (200, 200, 200), bottle_rect, 2)
        
        # Draw neck
        neck_rect = pygame.Rect(
            screen_pos[0] - screen_radius * 0.4,
            screen_pos[1] - screen_radius * 1.8,
            screen_radius * 0.8,
            screen_radius * 0.5
        )
        pygame.draw.rect(canvas, (200, 200, 200), neck_rect, 2)
    
    # Draw essence circle (split if combined)
    if state.is_combined:
        # Draw as pie slices
        n_parts = len(state.essence_types)
        angle_per_part = 360 / n_parts
        
        for i, essence_type in enumerate(state.essence_types):
            color = ESSENCE_TYPES[essence_type][1]
            
            # Draw pie slice
            start_angle = i * angle_per_part
            end_angle = (i + 1) * angle_per_part
            
            # Draw filled pie slice
            points = [screen_pos]
            for angle in range(int(start_angle), int(end_angle) + 1, 5):
                rad = np.deg2rad(angle)
                x = screen_pos[0] + radius * np.cos(rad)
                y = screen_pos[1] + radius * np.sin(rad)
                points.append((int(x), int(y)))
            points.append(screen_pos)
            
            pygame.draw.polygon(canvas, color, points)
            
            # Draw patterns for this slice if needed
            if state.enchanted_per_essence[i]:
                _draw_stripes_in_slice(canvas, screen_pos, radius, start_angle, end_angle, color)
            
            if state.refined_per_essence[i]:
                _draw_dots_in_slice(canvas, screen_pos, radius, start_angle, end_angle, color)
        
        # No outline for combined essences
    else:
        # Single essence - draw as simple circle
        color = ESSENCE_TYPES[state.essence_types[0]][1]
        pygame.draw.circle(canvas, color, screen_pos, screen_radius)
        
        # Draw patterns
        if state.enchanted_per_essence[0]:
            _draw_stripes(canvas, screen_pos, screen_radius, color)
        
        if state.refined_per_essence[0]:
            _draw_dots(canvas, screen_pos, screen_radius, color)
        
        # Draw outline
        pygame.draw.circle(canvas, (50, 50, 50), screen_pos, screen_radius, 2)


def _draw_stripes(canvas: pygame.Surface, pos: Tuple[int, int], radius: int, base_color: Tuple[int, int, int]):
    """Draw diagonal stripe pattern on a circle (enchanted)."""
    # Create a darker stripe color
    stripe_color = tuple(max(0, int(c * 0.6)) for c in base_color)
    
    # Draw diagonal lines clipped to circle
    spacing = max(3, radius // 5)
    line_width = max(1, radius // 15)
    
    for offset in range(-radius * 2, radius * 2, spacing):
        for y in range(-radius, radius + 1):
            x = y + offset
            # Check if point is within circle
            if x >= -radius and x <= radius:
                dist_sq = x*x + y*y
                if dist_sq <= radius * radius:
                    px = pos[0] + x
                    py = pos[1] + y
                    if abs(x - y - offset) < line_width:
                        canvas.set_at((int(px), int(py)), stripe_color)


def _draw_dots(canvas: pygame.Surface, pos: Tuple[int, int], radius: int, base_color: Tuple[int, int, int]):
    """Draw dotted pattern on a circle (refined)."""
    # Create a lighter dot color
    dot_color = tuple(min(255, int(c * 1.3)) for c in base_color)
    
    # Draw dots in a grid pattern, only within circle bounds
    dot_radius = max(1, radius // 12)
    spacing = max(3, radius // 4)
    
    for x_offset in range(-radius + spacing//2, radius, spacing):
        for y_offset in range(-radius + spacing//2, radius, spacing):
            # Only draw if within circle
            dist = np.sqrt(x_offset**2 + y_offset**2)
            if dist <= radius - dot_radius - 2:
                dot_x = int(pos[0] + x_offset)
                dot_y = int(pos[1] + y_offset)
                pygame.draw.circle(canvas, dot_color, (dot_x, dot_y), dot_radius)


def _draw_stripes_in_slice(
    canvas: pygame.Surface, 
    center: Tuple[int, int], 
    radius: float,
    start_angle: float, 
    end_angle: float,
    base_color: Tuple[int, int, int]
):
    """Draw stripes within a pie slice."""
    stripe_color = tuple(max(0, c - 60) for c in base_color)
    
    # Draw radial lines
    for angle in range(int(start_angle), int(end_angle), 15):
        rad = np.deg2rad(angle)
        x = center[0] + radius * np.cos(rad)
        y = center[1] + radius * np.sin(rad)
        pygame.draw.line(canvas, stripe_color, center, (int(x), int(y)), 2)


def _draw_dots_in_slice(
    canvas: pygame.Surface,
    center: Tuple[int, int],
    radius: float,
    start_angle: float,
    end_angle: float,
    base_color: Tuple[int, int, int]
):
    """Draw dots within a pie slice."""
    dot_color = tuple(min(255, c + 60) for c in base_color)
    dot_radius = max(1, int(radius / 8))
    
    mid_angle = (start_angle + end_angle) / 2
    rad = np.deg2rad(mid_angle)
    
    # Draw dots along the slice
    for r in range(dot_radius * 2, int(radius), int(radius / 3)):
        x = center[0] + r * np.cos(rad)
        y = center[1] + r * np.sin(rad)
        pygame.draw.circle(canvas, dot_color, (int(x), int(y)), dot_radius)


def draw_tool(canvas: pygame.Surface, tool, tile_size: float):
    """Draw a tool with its current state visualization."""
    from .entities import Cauldron, CauldronState
    
    pos = tool.position
    size = tool.size
    
    # Draw tool background
    rect = pygame.Rect(
        int(pos[0] - size[0] / 2),
        int(pos[1] - size[1] / 2),
        int(size[0]),
        int(size[1])
    )
    
    # Color based on state
    color = tool.color
    if hasattr(tool, 'state'):
        if tool.state == ToolState.PROCESSING or tool.state == CauldronState.STIRRING:
            # Pulse effect - make it brighter
            color = tuple(min(255, c + 40) for c in color)
        elif tool.state == ToolState.DONE or tool.state == CauldronState.DONE:
            # Flash green when done
            color = (100, 255, 100)
        elif tool.state == CauldronState.READY_TO_STIR:
            # Ready to stir - slight glow
            color = tuple(min(255, c + 20) for c in color)
    
    pygame.draw.rect(canvas, color, rect)
    pygame.draw.rect(canvas, (50, 50, 50), rect, 3)
    
    # Special rendering for Cauldron - show essences in slots
    if isinstance(tool, Cauldron):
        _draw_cauldron_contents(canvas, tool, tile_size)
    
    # Draw tool name
    font = pygame.font.SysFont('Arial', 10)
    text = font.render(tool.get_display_name(), True, (255, 255, 255))
    text_rect = text.get_rect(center=(int(pos[0]), int(pos[1])))
    canvas.blit(text, text_rect)


def _draw_cauldron_contents(canvas: pygame.Surface, cauldron, tile_size: float):
    """Draw essences in the cauldron's 4 slots and stir progress."""
    pos = cauldron.position
    slot_offset = tile_size * 0.5  # Reduced from 0.6 to keep patterns visible
    essence_radius = int(tile_size * 0.22)
    
    # Positions for 4 slots: top, right, bottom, left
    slot_positions = [
        (pos[0], pos[1] - slot_offset),  # Top
        (pos[0] + slot_offset, pos[1]),  # Right
        (pos[0], pos[1] + slot_offset),  # Bottom
        (pos[0] - slot_offset, pos[1]),  # Left
    ]
    
    # Draw essences in slots with patterns
    for i, essence_state in enumerate(cauldron.essence_slots):
        if essence_state is not None:
            slot_pos = slot_positions[i]
            from .entities import ESSENCE_TYPES
            
            # Check if this is a combined essence (multiple types)
            if essence_state.is_combined and len(essence_state.essence_types) > 1:
                # Draw as pie chart like in main game
                num_types = len(essence_state.essence_types)
                angle_per_type = 360 / num_types
                
                for j, essence_type in enumerate(essence_state.essence_types):
                    color = ESSENCE_TYPES[essence_type][1]
                    start_angle = j * angle_per_type - 90  # -90 to start from top
                    end_angle = (j + 1) * angle_per_type - 90
                    
                    # Draw pie slice
                    points = [(int(slot_pos[0]), int(slot_pos[1]))]
                    for angle in range(int(start_angle), int(end_angle) + 1, 5):
                        rad = np.deg2rad(angle)
                        x = slot_pos[0] + essence_radius * np.cos(rad)
                        y = slot_pos[1] + essence_radius * np.sin(rad)
                        points.append((int(x), int(y)))
                    
                    if len(points) > 2:
                        pygame.draw.polygon(canvas, color, points)
                    
                    # Draw patterns for this slice if needed
                    if essence_state.enchanted_per_essence[j]:
                        _draw_stripes_in_slice(canvas, (int(slot_pos[0]), int(slot_pos[1])), 
                                             essence_radius, start_angle, end_angle, color)
                    
                    if essence_state.refined_per_essence[j]:
                        _draw_dots_in_slice(canvas, (int(slot_pos[0]), int(slot_pos[1])), 
                                          essence_radius, start_angle, end_angle, color)
                
                # No outline for combined essences
            else:
                # Single essence - draw as simple circle
                color = ESSENCE_TYPES[essence_state.essence_types[0]][1]
                pygame.draw.circle(canvas, color, (int(slot_pos[0]), int(slot_pos[1])), essence_radius)
                
                # Draw patterns if enchanted/refined
                if essence_state.enchanted_per_essence[0]:
                    _draw_stripes(canvas, (int(slot_pos[0]), int(slot_pos[1])), essence_radius, color)
                if essence_state.refined_per_essence[0]:
                    _draw_dots(canvas, (int(slot_pos[0]), int(slot_pos[1])), essence_radius, color)
                
                # Outline for single essences
                pygame.draw.circle(canvas, (50, 50, 50), (int(slot_pos[0]), int(slot_pos[1])), essence_radius, 1)
    
    # Draw stir progress bar if stirring
    if cauldron.state == CauldronState.STIRRING and cauldron.stir_time > 0:
        progress = cauldron.stir_progress / cauldron.stir_time
        bar_width = int(cauldron.size[0] * 0.8)
        bar_height = 6
        bar_x = int(pos[0] - bar_width / 2)
        bar_y = int(pos[1] + cauldron.size[1] / 2 - 15)
        
        # Background
        pygame.draw.rect(canvas, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
        # Progress
        progress_width = int(bar_width * progress)
        pygame.draw.rect(canvas, (100, 255, 100), (bar_x, bar_y, progress_width, bar_height))
        # Border
        pygame.draw.rect(canvas, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), 1)


def draw_player(canvas: pygame.Surface, player: Player):
    """Draw the player character."""
    pos = player.body.position
    radius = player.radius
    
    screen_pos = (int(pos.x), int(pos.y))
    screen_radius = int(radius)
    
    # Draw player circle
    pygame.draw.circle(canvas, player.color, screen_pos, screen_radius)
    pygame.draw.circle(canvas, (30, 30, 30), screen_pos, screen_radius, 3)
    
    # Draw direction indicator
    vel = player.body.velocity
    if vel.length > 0.1:
        direction = vel.normalized() * radius * 0.8
        end_pos = (int(pos.x + direction.x), int(pos.y + direction.y))
        pygame.draw.line(canvas, (255, 255, 255), screen_pos, end_pos, 3)


def draw_dispenser(canvas: pygame.Surface, dispenser: Dispenser):
    """Draw a dispenser."""
    pos = dispenser.position
    size = (dispenser.tile_size * 0.8, dispenser.tile_size * 0.8)
    
    rect = pygame.Rect(
        int(pos[0] - size[0] / 2),
        int(pos[1] - size[1] / 2),
        int(size[0]),
        int(size[1])
    )
    
    # Draw dispenser with essence color
    pygame.draw.rect(canvas, dispenser.color, rect)
    pygame.draw.rect(canvas, (30, 30, 30), rect, 3)
    
    # Draw cooldown indicator
    if dispenser.cooldown > 0:
        # Draw overlay
        alpha = int(255 * (dispenser.cooldown / dispenser.cooldown_duration))
        overlay = pygame.Surface(size)
        overlay.fill((0, 0, 0))
        overlay.set_alpha(alpha // 2)
        canvas.blit(overlay, rect.topleft)


def draw_ui(
    canvas: pygame.Surface,
    time_remaining: int,
    time_limit: int,
    requirements: List[dict],
    map_width: int,
    map_height: int
):
    """
    Draw the UI elements (timer and requirements).
    
    Args:
        canvas: Pygame surface to draw on
        time_remaining: Steps remaining in the round
        time_limit: Total time limit for the round
        requirements: List of requirement dictionaries
        map_width: Width of the map
        map_height: Height of the map
    """
    # Draw timer bar at the top
    bar_height = 30
    bar_width = map_width - 40
    bar_x = 20
    bar_y = 10
    
    # Background
    pygame.draw.rect(canvas, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
    
    # Progress
    progress = time_remaining / time_limit if time_limit > 0 else 0
    progress_width = int(bar_width * progress)
    color = (100, 255, 100) if progress > 0.5 else (255, 200, 100) if progress > 0.25 else (255, 100, 100)
    pygame.draw.rect(canvas, color, (bar_x, bar_y, progress_width, bar_height))
    
    # Border
    pygame.draw.rect(canvas, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), 3)
    
    # Time text
    font = pygame.font.SysFont('Arial', 14)
    time_text = f"{time_remaining} / {time_limit}"
    text_surface = font.render(time_text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2))
    canvas.blit(text_surface, text_rect)
    
    # Draw requirements at the bottom
    req_y = map_height - 60
    req_x = 20
    
    font_small = pygame.font.SysFont('Arial', 12)
    
    for i, req in enumerate(requirements):
        # Draw requirement box
        box_size = 40
        box_x = req_x + i * (box_size + 10)
        
        # Background color based on completion
        bg_color = (100, 255, 100) if req.get('completed', False) else (150, 150, 150)
        pygame.draw.rect(canvas, bg_color, (box_x, req_y, box_size, box_size))
        pygame.draw.rect(canvas, (50, 50, 50), (box_x, req_y, box_size, box_size), 2)
        
        # Draw simplified essence representation
        essence_types = req['base_essences']
        if len(essence_types) == 1:
            color = ESSENCE_TYPES[essence_types[0]][1]
            center = (box_x + box_size // 2, req_y + box_size // 2)
            pygame.draw.circle(canvas, color, center, box_size // 3)
        else:
            # Multiple essences - draw as segments
            n_parts = len(essence_types)
            segment_width = box_size // n_parts
            for j, etype in enumerate(essence_types):
                color = ESSENCE_TYPES[etype][1]
                seg_rect = pygame.Rect(box_x + j * segment_width, req_y, segment_width, box_size)
                pygame.draw.rect(canvas, color, seg_rect)
        
        # Checkmark if completed
        if req.get('completed', False):
            check_text = font_small.render('✓', True, (255, 255, 255))
            canvas.blit(check_text, (box_x + 5, req_y + 5))


# ============================================================================
# Round Management
# ============================================================================

class RoundManager:
    """Manages rounds and their configurations."""
    
    def __init__(self, rounds_config: List[dict]):
        """
        Initialize with a list of round configurations.
        
        Args:
            rounds_config: List of round dictionaries
        """
        self.rounds = rounds_config
        self.current_round_index = 0
    
    def get_current_round(self) -> Optional[dict]:
        """Get the current round configuration."""
        if self.current_round_index < len(self.rounds):
            return self.rounds[self.current_round_index]
        return None
    
    def advance_round(self):
        """Move to the next round."""
        self.current_round_index += 1
    
    def reset(self):
        """Reset to the first round."""
        self.current_round_index = 0
    
    def is_complete(self) -> bool:
        """Check if all rounds are complete."""
        return self.current_round_index >= len(self.rounds)
    
    @staticmethod
    def load_from_file(filepath: str) -> 'RoundManager':
        """Load rounds from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        rounds = data.get('rounds', [])
        return RoundManager(rounds)
    
    @staticmethod
    def create_default_rounds() -> 'RoundManager':
        """Create a default set of rounds for testing."""
        rounds = [
            # Round 1: Deliver raw Fire Essence
            {
                "time_limit": 3600,
                "required_items": [
                    {
                        "base_essences": [1],
                        "enchanted": [False],
                        "refined": [False],
                        "bottled": False
                    }
                ],
                "description": "Deliver raw Fire Essence"
            },
            # Round 2: Deliver enchanted Fire Essence
            {
                "time_limit": 3600,
                "required_items": [
                    {
                        "base_essences": [1],
                        "enchanted": [True],
                        "refined": [False],
                        "bottled": False
                    }
                ],
                "description": "Deliver enchanted Fire Essence"
            },
            # Round 3: Deliver refined Water Essence
            {
                "time_limit": 3600,
                "required_items": [
                    {
                        "base_essences": [2],
                        "enchanted": [False],
                        "refined": [True],
                        "bottled": False
                    }
                ],
                "description": "Deliver refined Water Essence"
            },
            # Round 4: Deliver bottled Fire Essence
            {
                "time_limit": 3600,
                "required_items": [
                    {
                        "base_essences": [1],
                        "enchanted": [False],
                        "refined": [False],
                        "bottled": True
                    }
                ],
                "description": "Deliver bottled Fire Essence"
            },
            # Round 5: Deliver Fire+Water combination (both plain), bottled
            {
                "time_limit": 5400,
                "required_items": [
                    {
                        "base_essences": [1, 2],
                        "enchanted": [False, False],
                        "refined": [False, False],
                        "bottled": True
                    }
                ],
                "description": "Deliver bottled Fire+Water combination"
            },
            # Round 6: Deliver Fire(enchanted) + Water(plain), bottled
            {
                "time_limit": 5400,
                "required_items": [
                    {
                        "base_essences": [1, 2],
                        "enchanted": [True, False],
                        "refined": [False, False],
                        "bottled": True
                    }
                ],
                "description": "Deliver bottled Fire(enchanted)+Water(plain)"
            },
        ]
        
        return RoundManager(rounds)


# ============================================================================
# Layout Management
# ============================================================================

def create_default_layout(
    space: pymunk.Space,
    map_width: float,
    map_height: float,
    tile_size: float
) -> Dict[str, any]:
    """
    Create the default laboratory layout with tools and dispensers.
    
    Returns a dictionary containing all game objects.
    """
    # Calculate positions
    margin = tile_size * 2
    top_y = margin
    mid_y = map_height / 2
    bottom_y = map_height - margin
    
    # Create dispensers at the top (4 dispensers for Fire, Water, Earth, Air)
    dispensers = []
    dispenser_spacing = map_width / 5
    for i, essence_type in enumerate([1, 2, 3, 4]):  # Fire, Water, Earth, Air
        x = margin + (i + 1) * dispenser_spacing
        dispenser = Dispenser(space, (x, top_y), essence_type, tile_size)
        dispensers.append(dispenser)
    
    # Create tools in the middle area
    enchanter = Enchanter(space, (margin + tile_size * 3, mid_y - tile_size * 2), tile_size)
    refiner = Refiner(space, (map_width - margin - tile_size * 3, mid_y - tile_size * 2), tile_size)
    cauldron = Cauldron(space, (map_width / 2, mid_y), tile_size)
    bottler = Bottler(space, (map_width / 2 + tile_size * 3, mid_y + tile_size * 2), tile_size)
    trash_can = TrashCan(space, (map_width - margin - tile_size, mid_y - tile_size * 2), tile_size)
    
    # Create delivery window at the bottom
    delivery_window = DeliveryWindow(space, (map_width / 2, bottom_y), tile_size)
    
    # Create player
    player_start_pos = (map_width / 2 + tile_size * 4, mid_y + tile_size * 3)
    player = Player(space, player_start_pos, tile_size)
    
    return {
        'dispensers': dispensers,
        'enchanter': enchanter,
        'refiner': refiner,
        'cauldron': cauldron,
        'bottler': bottler,
        'trash_can': trash_can,
        'delivery_window': delivery_window,
        'player': player,
    }

