"""
Game logic for Potion Lab environment.

Contains:
- Physics setup and configuration
- Collision handlers for player-tool interactions
- Rendering functions for essences with visual patterns
- Round configuration and management
"""

import json
import os

import numpy as np
import pygame
import pymunk
from pymunk.vec2d import Vec2d

from .entities import (
    ESSENCE_TYPES,
    Bottler,
    Cauldron,
    CauldronState,
    DeliveryWindow,
    Dispenser,
    Enchanter,
    Essence,
    Player,
    Refiner,
    ToolState,
    TrashCan,
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
    LAYER_DISPENSER = 5
    LAYER_CAULDRON = 6

    # Physics constants
    GRAVITY = (0, 0)
    DAMPING = 0.8  # Objects slow down naturally (higher = less damping, 0.8 = 20% velocity loss per second)
    ITERATIONS = 20  # Higher iterations = better collision resolution
    TIMESTEP = 1 / 60  # 60 Hz


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
    """Manages collision detection and response between game objects using pymunk's collision handler system."""

    def __init__(self, env):
        """
        Initialize collision handler.

        Args:
            env: Reference to the PotionLabEnv
        """
        self.env = env
        self.player_stirring_cauldron = False  # Track if player is currently stirring
        self.essence_tool_collisions = set()  # Track active essence-tool collisions

    def setup_handlers(self, space: pymunk.Space):
        """
        Set up pymunk collision handlers for game interactions.

        Args:
            space: The pymunk physics space
        """
        # Player-Dispenser collisions (trigger essence dispensing)
        space.on_collision(
            collision_type_a=PhysicsConfig.LAYER_PLAYER,
            collision_type_b=PhysicsConfig.LAYER_DISPENSER,
            begin=self._on_player_dispenser_begin,
        )

        # Player-Cauldron collisions (for stirring)
        space.on_collision(
            collision_type_a=PhysicsConfig.LAYER_PLAYER,
            collision_type_b=PhysicsConfig.LAYER_CAULDRON,
            begin=self._on_player_cauldron_begin,
            separate=self._on_player_cauldron_separate,
        )

        # Essence-Tool collisions (for processing)
        space.on_collision(
            collision_type_a=PhysicsConfig.LAYER_ESSENCE,
            collision_type_b=PhysicsConfig.LAYER_TOOL,
            begin=self._on_essence_tool_begin,
        )

        # Essence-Cauldron collisions (cauldron has special collision type)
        space.on_collision(
            collision_type_a=PhysicsConfig.LAYER_ESSENCE,
            collision_type_b=PhysicsConfig.LAYER_CAULDRON,
            begin=self._on_essence_cauldron_begin,
        )

    def update(self):
        """Update collision state each frame."""
        # Handle ongoing cauldron stirring
        if self.player_stirring_cauldron:
            cauldron = self.env.tools.get("cauldron")
            if cauldron:
                cauldron.stir()

    def _on_player_dispenser_begin(self, arbiter, space, data):
        """Called when player collides with a dispenser."""
        # Get the dispenser from the collision
        for shape in arbiter.shapes:
            if hasattr(shape, "dispenser_obj"):
                dispenser = shape.dispenser_obj
                essence_state = dispenser.dispense()

                if essence_state is not None:
                    # Spawn essence at dispenser location
                    position = dispenser.position
                    # Offset slightly so player can push it away
                    offset = Vec2d(0, dispenser.tile_size * 1.25)
                    spawn_pos = (position[0] + offset.x, position[1] + offset.y)

                    # Get essence physics properties from variation space
                    mass = self.env.variation_space["essence"]["mass"].value
                    friction = self.env.variation_space["essence"]["friction"].value
                    elasticity = self.env.variation_space["essence"]["elasticity"].value

                    essence = Essence(
                        self.env.space, spawn_pos, essence_state, self.env.tile_size, mass, friction, elasticity
                    )
                    self.env.essences.append(essence)
                break

    def _on_player_cauldron_begin(self, arbiter, space, data):
        """Called when player starts colliding with cauldron."""
        cauldron = self.env.tools.get("cauldron")
        if cauldron:
            cauldron.start_stirring()
            self.player_stirring_cauldron = True

    def _on_player_cauldron_separate(self, arbiter, space, data):
        """Called when player stops colliding with cauldron."""
        cauldron = self.env.tools.get("cauldron")
        if cauldron:
            cauldron.stop_stirring()
            self.player_stirring_cauldron = False

    def _on_essence_tool_begin(self, arbiter, space, data):
        """Called when essence collides with a tool."""
        # Find essence and tool from collision
        essence_obj = None
        tool_obj = None
        tool_name = None

        for shape in arbiter.shapes:
            if hasattr(shape, "essence_obj"):
                essence_obj = shape.essence_obj
            elif hasattr(shape, "tool_obj"):
                tool_obj = shape.tool_obj
                # Find the tool name
                for name, tool in self.env.tools.items():
                    if tool is tool_obj:
                        tool_name = name
                        break

        if essence_obj is None or tool_obj is None:
            return

        # Create collision key to ensure we only process once
        collision_key = (id(essence_obj), id(tool_obj))
        if collision_key in self.essence_tool_collisions:
            return

        self.essence_tool_collisions.add(collision_key)

        # Handle delivery window specially
        if tool_name == "delivery_window":
            if hasattr(tool_obj, "validate_delivery"):
                accepted = tool_obj.validate_delivery(essence_obj)
                if accepted and essence_obj in self.env.essences:
                    self.env.essences.remove(essence_obj)
        # Try to accept the essence for other tools
        elif hasattr(tool_obj, "accept_essence"):
            accepted = tool_obj.accept_essence(essence_obj)
            if accepted and essence_obj in self.env.essences:
                self.env.essences.remove(essence_obj)

    def _on_essence_cauldron_begin(self, arbiter, space, data):
        """Called when essence collides with the cauldron."""
        # Find essence from collision
        essence_obj = None

        for shape in arbiter.shapes:
            if hasattr(shape, "essence_obj"):
                essence_obj = shape.essence_obj
                break

        if essence_obj is None:
            return

        # Get cauldron
        cauldron = self.env.tools.get("cauldron")
        if cauldron is None:
            return

        # Create collision key to ensure we only process once
        collision_key = (id(essence_obj), id(cauldron))
        if collision_key in self.essence_tool_collisions:
            return

        self.essence_tool_collisions.add(collision_key)

        # Try to add essence to cauldron
        if hasattr(cauldron, "accept_essence"):
            accepted = cauldron.accept_essence(essence_obj)
            if accepted and essence_obj in self.env.essences:
                self.env.essences.remove(essence_obj)


# ============================================================================
# Rendering Functions
# ============================================================================


def draw_essence(canvas: pygame.Surface, essence: Essence, tile_size: float):
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

    screen_pos = (int(pos.x), int(pos.y))
    screen_radius = int(radius)

    if state.is_bottled:
        bottle_rect = pygame.Rect(
            screen_pos[0] - screen_radius * 1.2,
            screen_pos[1] - screen_radius * 1.5,
            screen_radius * 2.4,
            screen_radius * 3,
        )
        pygame.draw.rect(canvas, (200, 200, 200), bottle_rect, 2)

        neck_rect = pygame.Rect(
            screen_pos[0] - screen_radius * 0.4,
            screen_pos[1] - screen_radius * 1.8,
            screen_radius * 0.8,
            screen_radius * 0.5,
        )
        pygame.draw.rect(canvas, (200, 200, 200), neck_rect, 2)

    if state.is_combined:
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
                _draw_stripes_in_slice(canvas, screen_pos, radius, start_angle, end_angle)

            if state.refined_per_essence[i]:
                _draw_dots_in_slice(canvas, screen_pos, radius, start_angle, end_angle)

    else:
        color = ESSENCE_TYPES[state.essence_types[0]][1]
        pygame.draw.circle(canvas, color, screen_pos, screen_radius)

        # Draw patterns
        if state.enchanted_per_essence[0]:
            _draw_stripes(canvas, screen_pos, screen_radius)

        if state.refined_per_essence[0]:
            _draw_dots(canvas, screen_pos, screen_radius)


def _draw_stripes(canvas: pygame.Surface, pos: tuple[int, int], radius: int):
    """Draw diagonal stripe pattern on a circle (enchanted)."""
    stripe_color = (0, 0, 0)

    # Draw diagonal lines clipped to circle
    spacing = max(3, radius // 5)
    line_width = max(1, radius // 15)

    for offset in range(-radius * 2, radius * 2, spacing):
        for y in range(-radius, radius + 1):
            x = y + offset
            # Check if point is within circle
            if x >= -radius and x <= radius:
                dist_sq = x * x + y * y
                if dist_sq <= radius * radius:
                    px = pos[0] + x
                    py = pos[1] + y
                    if abs(x - y - offset) < line_width:
                        canvas.set_at((int(px), int(py)), stripe_color)


def _draw_dots(canvas: pygame.Surface, pos: tuple[int, int], radius: int):
    """Draw dotted pattern on a circle (refined)."""
    dot_color = (0, 0, 0)

    # Draw dots in a grid pattern, only within circle bounds
    dot_radius = max(1, radius // 12)
    spacing = max(3, radius // 4)

    for x_offset in range(-radius + spacing // 2, radius, spacing):
        for y_offset in range(-radius + spacing // 2, radius, spacing):
            # Only draw if within circle
            dist = np.sqrt(x_offset**2 + y_offset**2)
            if dist <= radius - dot_radius - 2:
                dot_x = int(pos[0] + x_offset)
                dot_y = int(pos[1] + y_offset)
                pygame.draw.circle(canvas, dot_color, (dot_x, dot_y), dot_radius)


def _draw_stripes_in_slice(
    canvas: pygame.Surface,
    center: tuple[int, int],
    radius: float,
    start_angle: float,
    end_angle: float,
):
    """Draw diagonal stripes within a pie slice - looks like spliced top/bottom."""
    stripe_color = (0, 0, 0)

    # Draw diagonal stripes clipped to the pie slice
    # The stripes are diagonal (same as full circle), just clipped to the slice
    spacing = max(3, int(radius) // 5)
    line_width = max(1, int(radius) // 15)

    # Create list of points that define the pie slice
    slice_points = [(int(center[0]), int(center[1]))]
    for angle in range(int(start_angle), int(end_angle) + 1, 2):
        rad = np.deg2rad(angle)
        x = center[0] + radius * np.cos(rad)
        y = center[1] + radius * np.sin(rad)
        slice_points.append((int(x), int(y)))
    slice_points.append((int(center[0]), int(center[1])))

    # Draw diagonal stripes only within the pie slice
    for offset in range(-int(radius) * 2, int(radius) * 2, spacing):
        for y in range(-int(radius), int(radius) + 1):
            x = y + offset
            # Check if point is within circle
            if x >= -radius and x <= radius:
                dist_sq = x * x + y * y
                if dist_sq <= radius * radius:
                    px = int(center[0] + x)
                    py = int(center[1] + y)

                    # Check if point is within the pie slice using angle
                    angle_to_point = np.rad2deg(np.arctan2(y, x)) % 360
                    if start_angle <= angle_to_point <= end_angle or (
                        end_angle < start_angle and (angle_to_point >= start_angle or angle_to_point <= end_angle)
                    ):
                        if abs(x - y - offset) < line_width:
                            canvas.set_at((px, py), stripe_color)


def _draw_dots_in_slice(
    canvas: pygame.Surface,
    center: tuple[int, int],
    radius: float,
    start_angle: float,
    end_angle: float,
):
    """Draw dots within a pie slice - looks like spliced top/bottom."""
    # Use black for dots
    dot_color = (0, 0, 0)
    dot_radius = max(1, int(radius) // 12)
    spacing = max(3, int(radius) // 4)

    # Draw dots in grid pattern, only within the pie slice
    for x_offset in range(-int(radius) + spacing // 2, int(radius), spacing):
        for y_offset in range(-int(radius) + spacing // 2, int(radius), spacing):
            # Check if within circle
            dist = np.sqrt(x_offset**2 + y_offset**2)
            if dist <= radius - dot_radius - 2:
                # Check if point is within the pie slice using angle
                angle_to_point = np.rad2deg(np.arctan2(y_offset, x_offset)) % 360
                if start_angle <= angle_to_point <= end_angle or (
                    end_angle < start_angle and (angle_to_point >= start_angle or angle_to_point <= end_angle)
                ):
                    dot_x = int(center[0] + x_offset)
                    dot_y = int(center[1] + y_offset)
                    pygame.draw.circle(canvas, dot_color, (dot_x, dot_y), dot_radius)


def draw_tool(canvas: pygame.Surface, tool, tile_size: float):
    """Draw a tool with its current state visualization."""
    pos = tool.position
    size = tool.size

    rect = pygame.Rect(int(pos[0] - size[0] / 2), int(pos[1] - size[1] / 2), int(size[0]), int(size[1]))

    color = tool.color
    if hasattr(tool, "state"):
        if tool.state == ToolState.PROCESSING or tool.state == CauldronState.STIRRING:
            color = tuple(min(255, c + 40) for c in color)
        elif tool.state == ToolState.DONE or tool.state == CauldronState.DONE:
            color = (100, 255, 100)
        elif tool.state == CauldronState.READY_TO_STIR:
            color = tuple(min(255, c + 20) for c in color)

    pygame.draw.rect(canvas, color, rect)
    pygame.draw.rect(canvas, (50, 50, 50), rect, 3)

    # Special rendering for Cauldron - show essences in slots
    if isinstance(tool, Cauldron):
        _draw_cauldron_contents(canvas, tool, tile_size)

    font = pygame.font.SysFont("Arial", 10)
    text = font.render(tool.get_display_name(), True, (255, 255, 255))
    text_rect = text.get_rect(center=(int(pos[0]), int(pos[1])))
    canvas.blit(text, text_rect)


def _draw_cauldron_contents(canvas: pygame.Surface, cauldron, tile_size: float):
    """Draw essences in the cauldron's 4 slots and stir progress."""
    pos = cauldron.position
    slot_offset = tile_size * 0.6  # Increased to accommodate larger essences
    essence_radius = int(tile_size * 0.35)  # Same size as floor essences

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
            if essence_state.is_combined and len(essence_state.essence_types) > 1:
                num_types = len(essence_state.essence_types)
                angle_per_type = 360 / num_types

                for j, essence_type in enumerate(essence_state.essence_types):
                    color = ESSENCE_TYPES[essence_type][1]
                    start_angle = j * angle_per_type  # Same orientation as ground essences
                    end_angle = (j + 1) * angle_per_type

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
                        _draw_stripes_in_slice(
                            canvas, (int(slot_pos[0]), int(slot_pos[1])), essence_radius, start_angle, end_angle
                        )

                    if essence_state.refined_per_essence[j]:
                        _draw_dots_in_slice(
                            canvas, (int(slot_pos[0]), int(slot_pos[1])), essence_radius, start_angle, end_angle
                        )

            else:
                # Single essence - draw as simple circle
                color = ESSENCE_TYPES[essence_state.essence_types[0]][1]
                pygame.draw.circle(canvas, color, (int(slot_pos[0]), int(slot_pos[1])), essence_radius)

                # Draw patterns if enchanted/refined
                if essence_state.enchanted_per_essence[0]:
                    _draw_stripes(canvas, (int(slot_pos[0]), int(slot_pos[1])), essence_radius)
                if essence_state.refined_per_essence[0]:
                    _draw_dots(canvas, (int(slot_pos[0]), int(slot_pos[1])), essence_radius)

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
    half_size = player.size / 2

    # Get square vertices
    vertices = [
        (int(pos.x - half_size), int(pos.y - half_size)),
        (int(pos.x + half_size), int(pos.y - half_size)),
        (int(pos.x + half_size), int(pos.y + half_size)),
        (int(pos.x - half_size), int(pos.y + half_size)),
    ]

    # Draw player square
    pygame.draw.polygon(canvas, player.color, vertices)
    pygame.draw.polygon(canvas, (30, 30, 30), vertices, 3)

    # Draw direction indicator
    vel = player.body.velocity
    if vel.length > 0.1:
        screen_pos = (int(pos.x), int(pos.y))
        direction = vel.normalized() * half_size * 0.8
        end_pos = (int(pos.x + direction.x), int(pos.y + direction.y))
        pygame.draw.line(canvas, (255, 255, 255), screen_pos, end_pos, 3)


def draw_dispenser(canvas: pygame.Surface, dispenser: Dispenser):
    """Draw a dispenser."""
    pos = dispenser.position
    size = (dispenser.tile_size * 0.8, dispenser.tile_size * 0.8)

    rect = pygame.Rect(int(pos[0] - size[0] / 2), int(pos[1] - size[1] / 2), int(size[0]), int(size[1]))

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
    requirements: list[dict],
    map_width: int,
    map_height: int,
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
    font = pygame.font.SysFont("Arial", 14)
    time_text = f"{time_remaining} / {time_limit}"
    text_surface = font.render(time_text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2))
    canvas.blit(text_surface, text_rect)

    # Draw requirements at the bottom
    req_y = map_height - 60
    req_x = 20

    for i, req in enumerate(requirements):
        # Draw requirement box
        box_size = 40
        box_x = req_x + i * (box_size + 10)

        # Background color based on completion
        bg_color = (100, 255, 100) if req.get("completed", False) else (150, 150, 150)
        pygame.draw.rect(canvas, bg_color, (box_x, req_y, box_size, box_size))
        pygame.draw.rect(canvas, (50, 50, 50), (box_x, req_y, box_size, box_size), 2)

        essence_types = req["base_essences"]
        if len(essence_types) == 1:
            color = ESSENCE_TYPES[essence_types[0]][1]
            center = (box_x + box_size // 2, req_y + box_size // 2)
            pygame.draw.circle(canvas, color, center, box_size // 3)
        else:
            n_parts = len(essence_types)
            segment_width = box_size // n_parts
            for j, etype in enumerate(essence_types):
                color = ESSENCE_TYPES[etype][1]
                seg_rect = pygame.Rect(box_x + j * segment_width, req_y, segment_width, box_size)
                pygame.draw.rect(canvas, color, seg_rect)


# ============================================================================
# Round Management
# ============================================================================


class RoundManager:
    """Manages rounds and their configurations."""

    def __init__(self, rounds_config: list[dict]):
        """
        Initialize with a list of round configurations.

        Args:
            rounds_config: List of round dictionaries
        """
        self.rounds = rounds_config
        self.current_round_index = 0

    def get_current_round(self) -> dict | None:
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
    def load_from_file(filepath: str) -> "RoundManager":
        """Load rounds from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        rounds = data.get("rounds", [])
        return RoundManager(rounds)

    @staticmethod
    def create_default_rounds() -> "RoundManager":
        """Load default rounds from rounds.json file."""
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rounds_file = os.path.join(current_dir, "rounds.json")

        return RoundManager.load_from_file(rounds_file)


# ============================================================================
# Layout Management
# ============================================================================


def create_default_layout(
    space: pymunk.Space, map_width: float, map_height: float, tile_size: float, layout_file: str | None = None
) -> dict[str, any]:
    """
    Create the laboratory layout from a JSON file.

    Args:
        space: Pymunk physics space
        map_width: Width of the map in pixels (unused, kept for compatibility)
        map_height: Height of the map in pixels (unused, kept for compatibility)
        tile_size: Size of each tile in pixels
        layout_file: Path to the layout JSON file. If None, loads default layout.json

    Returns a dictionary containing all game objects.
    """
    # Load layout from JSON file
    if layout_file is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        layout_file = os.path.join(current_dir, "layout.json")

    with open(layout_file) as f:
        layout_data = json.load(f)

    # Create dispensers from layout data
    dispensers = []
    for dispenser_data in layout_data["dispensers"]:
        dispenser = Dispenser(
            space, (dispenser_data["x"], dispenser_data["y"]), dispenser_data["essence_type"], tile_size
        )
        dispensers.append(dispenser)

    # Create tools from layout data
    tools_data = layout_data["tools"]
    enchanter = Enchanter(space, (tools_data["enchanter"]["x"], tools_data["enchanter"]["y"]), tile_size)
    refiner = Refiner(space, (tools_data["refiner"]["x"], tools_data["refiner"]["y"]), tile_size)
    cauldron = Cauldron(space, (tools_data["cauldron"]["x"], tools_data["cauldron"]["y"]), tile_size)
    bottler = Bottler(space, (tools_data["bottler"]["x"], tools_data["bottler"]["y"]), tile_size)
    trash_can = TrashCan(space, (tools_data["trash_can"]["x"], tools_data["trash_can"]["y"]), tile_size)
    delivery_window = DeliveryWindow(
        space, (tools_data["delivery_window"]["x"], tools_data["delivery_window"]["y"]), tile_size
    )

    # Get player position from layout data (player creation happens in env)
    player_data = layout_data["player"]
    player_position = (player_data["x"], player_data["y"])

    return {
        "dispensers": dispensers,
        "enchanter": enchanter,
        "refiner": refiner,
        "cauldron": cauldron,
        "bottler": bottler,
        "trash_can": trash_can,
        "delivery_window": delivery_window,
        "player_position": player_position,
    }
