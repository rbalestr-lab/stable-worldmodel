"""
Potion Lab Environment - A physics-based continual learning environment for world models.

This environment tests compositional generalization, catastrophic forgetting, and
long-horizon planning through potion brewing mechanics.
"""

import json
import os
from collections.abc import Sequence
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

import stable_worldmodel as swm

# Import core game components
from .potion_lab_core import (
    ESSENCE_TYPES,
    CollisionHandler,
    Essence,
    PhysicsConfig,
    Player,
    RoundManager,
    add_walls,
    create_default_layout,
    draw_dispenser,
    draw_essence,
    draw_player,
    draw_tool,
    setup_physics_space,
)
from .potion_lab_core.game_logic import (
    _draw_dots,
    _draw_dots_in_slice,
    _draw_stripes,
    _draw_stripes_in_slice,
)


DEFAULT_VARIATIONS = (
    "player.start_position",
    "player.color",
    "player.size",
    "player.mass",
    "player.friction",
    "player.elasticity",
    "player.speed",
    "essence.mass",
    "essence.friction",
    "essence.elasticity",
    "background.color",
)


class PotionLab(gym.Env):
    """
    Potion Lab - A continual learning environment for world models.

    The agent must learn to brew potions by:
    1. Collecting essences from dispensers
    2. Processing them through various tools (Enchanter, Refiner, Cauldron, Bottler)
    3. Delivering completed potions to the delivery window

    Features:
    - Physics-based interactions using Pymunk
    - Progressive difficulty through 60+ rounds
    - Compositional generalization testing
    - Visual patterns for essence states (stripes=enchanted, dots=refined)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        resolution: int = 512,
        render_action: bool = False,
        layout_path: str | None = None,
        rounds_path: str | None = None,
    ):
        """
        Initialize the Potion Lab environment.

        Args:
            render_mode: "human" or "rgb_array"
            resolution: Render resolution (square image)
            render_action: Whether to render action indicators
            layout_path: Path to layout JSON file (default: potion_lab_core/layout.json)
            rounds_path: Path to rounds JSON file (default: potion_lab_core/rounds.json)
        """
        super().__init__()

        self._seed = None
        self.render_mode = render_mode
        self.render_size = resolution
        self.window_size = 512
        self.render_action = render_action

        # Config files (use defaults if not provided)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.layout_path = layout_path or os.path.join(current_dir, "potion_lab_core", "layout.json")
        self.rounds_path = rounds_path or os.path.join(current_dir, "potion_lab_core", "rounds.json")

        # Load UI bar heights
        with open(self.layout_path) as f:
            layout_data = json.load(f)

        ui_config = layout_data.get("ui", {})
        self.ui_top_height = ui_config.get("top_height", 50)
        self.ui_bottom_height = ui_config.get("bottom_height", 80)

        # Lab fills remaining space between UI bars
        self.map_width = self.window_size  # Full width: 512
        self.map_height = self.window_size - self.ui_top_height - self.ui_bottom_height  # Remaining height

        self.map_width_tiles = 16
        self.map_height_tiles = 16
        self.tile_size = self.map_width / self.map_width_tiles  # 32.0 if window_size=512

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8),
                "proprio": spaces.Box(
                    low=0,
                    high=self.window_size,
                    shape=(4,),  # player x, y, vx, vy
                    dtype=np.float32,
                ),
            }
        )

        self.variation_space = swm.spaces.Dict(
            {
                "player": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(init_value=np.array(pygame.Color("RoyalBlue")[:3], dtype=np.uint8)),
                        "size": swm.spaces.Box(
                            low=8.0,
                            high=16.0,
                            init_value=12.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "mass": swm.spaces.Box(
                            low=0.5,
                            high=3.0,
                            init_value=1.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "friction": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=0.3,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "elasticity": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=0.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "speed": swm.spaces.Box(
                            low=100.0,
                            high=300.0,
                            init_value=200.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "start_position": swm.spaces.Box(
                            low=50.0,
                            high=self.window_size - 50.0,
                            init_value=np.array([384.0, 352.0], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                        ),
                    }
                ),
                "essence": swm.spaces.Dict(
                    {
                        "mass": swm.spaces.Box(
                            low=0.1,
                            high=2.0,
                            init_value=0.5,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "friction": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=0.5,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "elasticity": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=0.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                "physics": swm.spaces.Dict(
                    {
                        "gravity": swm.spaces.Box(
                            low=-10.0,
                            high=10.0,
                            init_value=np.array([0.0, 0.0], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                        ),
                    }
                ),
                "background": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(init_value=np.array([40, 40, 45], dtype=np.uint8)),
                    }
                ),
            },
            sampling_order=["background", "physics", "essence", "player"],
        )

        # Pygame and rendering
        self.window = None
        self.clock = None
        self.canvas = None

        # Game state
        self.space = None
        self.player = None
        self.essences = []
        self.tools = {}
        self.dispensers = []
        self.collision_handler = None
        self.round_manager = None

        # Timing
        self.step_count = 0
        self.round_time_remaining = 0

        self.rng = None  # gets seeded in reset

        self.latest_action = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)

        self.rng = np.random.default_rng(seed)

        if hasattr(self, "variation_space"):
            self.variation_space.seed(seed)

        options = options or {}

        self.variation_space.reset()

        variations = options.get("variation", DEFAULT_VARIATIONS)

        if not isinstance(variations, Sequence):
            raise ValueError("variation option must be a Sequence containing variation names to sample")

        self.variation_space.update(variations)

        assert self.variation_space.check(debug=True), "Variation values must be within variation space!"

        self._setup()

        self.round_manager = RoundManager.load_from_file(self.rounds_path)

        round_config = self.round_manager.get_current_round()
        if round_config is None:
            raise RuntimeError("No rounds configured!")

        requirements = round_config["required_items"].copy()
        for req in requirements:
            req["completed"] = False
        self.tools["delivery_window"].set_requirements(requirements)

        self.round_time_remaining = round_config["time_limit"]
        self.step_count = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _setup(self):
        """Initialize physics space and game objects."""
        self.space = setup_physics_space()

        self.space.gravity = self.variation_space["physics"]["gravity"].value.tolist()

        add_walls(self.space, self.map_width, self.map_height)

        layout = create_default_layout(
            self.space, self.map_width, self.map_height, self.tile_size, layout_file=self.layout_path
        )

        # Create player with physics properties from variation space
        start_pos = self.variation_space["player"]["start_position"].value
        self.player = Player(
            self.space,
            position=(float(start_pos[0]), float(start_pos[1])),
            tile_size=self.tile_size,
            size=self.variation_space["player"]["size"].value,
            mass=self.variation_space["player"]["mass"].value,
            friction=self.variation_space["player"]["friction"].value,
            elasticity=self.variation_space["player"]["elasticity"].value,
            max_velocity=self.variation_space["player"]["speed"].value,
            color=tuple(self.variation_space["player"]["color"].value.tolist()),
        )

        self.tools = {
            "enchanter": layout["enchanter"],
            "refiner": layout["refiner"],
            "cauldron": layout["cauldron"],
            "bottler": layout["bottler"],
            "trash_can": layout["trash_can"],
            "delivery_window": layout["delivery_window"],
        }
        self.dispensers = layout["dispensers"]

        self.collision_handler = CollisionHandler(self)
        self.collision_handler.setup_handlers(self.space)

        self.essences = []

    def step(self, action: np.ndarray):
        """
        Execute one step in the environment.

        Args:
            action: [vx, vy] velocity commands in [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.latest_action = action

        self.player.apply_action(action)

        self.space.step(PhysicsConfig.TIMESTEP)

        self.collision_handler.update()

        for tool in self.tools.values():
            tool.update(PhysicsConfig.TIMESTEP)

        for dispenser in self.dispensers:
            dispenser.update(PhysicsConfig.TIMESTEP)

        self._check_tool_ejections()

        self.round_time_remaining -= 1
        self.step_count += 1

        terminated = False
        truncated = False
        reward = 0.0

        if self.tools["delivery_window"].all_requirements_met():
            # Round is complete!
            # Give reward here. Here and failure are the only rewards.
            reward = 1.0
            self.round_manager.advance_round()

            if self.round_manager.is_complete():
                terminated = True
            else:
                self._load_next_round()

        elif self.round_time_remaining <= 0:
            # Round failed. Give negative reward here.
            reward = -1.0
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _load_next_round(self):
        """Load the next round configuration without full reset."""
        round_config = self.round_manager.get_current_round()
        if round_config is None:
            return

        for essence in self.essences[:]:
            essence.remove_from_world()
        self.essences.clear()

        for tool in self.tools.values():
            tool.reset()

        if hasattr(self, "collision_handler"):
            self.collision_handler.player_stirring_cauldron = False

        requirements = round_config["required_items"].copy()
        for req in requirements:
            req["completed"] = False
        self.tools["delivery_window"].set_requirements(requirements)

        self.round_time_remaining = round_config["time_limit"]

    def _check_tool_ejections(self):
        """Check if any tools are ready to eject processed essences."""
        eject_offset = self.tile_size * 1.5  # Increased from 0.8 to 1.5

        # Get essence physics properties from variation space
        mass = self.variation_space["essence"]["mass"].value
        friction = self.variation_space["essence"]["friction"].value
        elasticity = self.variation_space["essence"]["elasticity"].value

        if hasattr(self.tools["enchanter"], "eject_essence"):
            essence_state = self.tools["enchanter"].eject_essence()
            if essence_state is not None:
                pos = (self.tools["enchanter"].position[0] + eject_offset, self.tools["enchanter"].position[1])
                essence = Essence(self.space, pos, essence_state, self.tile_size, mass, friction, elasticity)
                self.essences.append(essence)

        if hasattr(self.tools["refiner"], "eject_essence"):
            essence_state = self.tools["refiner"].eject_essence()
            if essence_state is not None:
                pos = (self.tools["refiner"].position[0] - eject_offset, self.tools["refiner"].position[1])
                essence = Essence(self.space, pos, essence_state, self.tile_size, mass, friction, elasticity)
                self.essences.append(essence)

        if hasattr(self.tools["cauldron"], "eject_essence"):
            essence_state = self.tools["cauldron"].eject_essence()
            if essence_state is not None:
                pos = (self.tools["cauldron"].position[0], self.tools["cauldron"].position[1] + eject_offset)
                essence = Essence(self.space, pos, essence_state, self.tile_size, mass, friction, elasticity)
                self.essences.append(essence)

        if hasattr(self.tools["bottler"], "eject_essence"):
            essence_state = self.tools["bottler"].eject_essence()
            if essence_state is not None:
                pos = (self.tools["bottler"].position[0], self.tools["bottler"].position[1] + eject_offset)
                essence = Essence(self.space, pos, essence_state, self.tile_size, mass, friction, elasticity)
                self.essences.append(essence)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Get the current observation."""
        img = self.render()

        # Proprioception vector
        player_pos = np.array([self.player.body.position.x, self.player.body.position.y], dtype=np.float32)
        player_vel = np.array([self.player.body.velocity.x, self.player.body.velocity.y], dtype=np.float32)
        proprio = np.concatenate([player_pos, player_vel])

        return {"image": img, "proprio": proprio}

    def _get_info(self) -> dict[str, Any]:
        """Get auxiliary information."""
        round_config = self.round_manager.get_current_round()

        return {
            "round_index": self.round_manager.current_round_index,
            "time_remaining": self.round_time_remaining,
            "time_limit": round_config["time_limit"] if round_config else 0,
            "requirements_met": self.tools["delivery_window"].all_requirements_met(),
            "num_essences": len(self.essences),
            "player_pos": np.array([self.player.body.position.x, self.player.body.position.y], dtype=np.float32),
            "player_vel": np.array([self.player.body.velocity.x, self.player.body.velocity.y], dtype=np.float32),
            "round_description": round_config.get("description", "") if round_config else "",
            "required_items": self.tools["delivery_window"].required_items,
        }

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self.canvas is None:
            pygame.init()
            # Canvas is always window_size x window_size (512x512)
            self.canvas = pygame.Surface((self.window_size, self.window_size))

            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("Potion Lab")
                self.clock = pygame.time.Clock()

        bg_color = tuple(self.variation_space["background"]["color"].value.tolist())
        ui_bg_color = (max(0, bg_color[0] - 10), max(0, bg_color[1] - 10), max(0, bg_color[2] - 10))

        self.canvas.fill(ui_bg_color)

        game_area_rect = pygame.Rect(0, self.ui_top_height, int(self.map_width), int(self.map_height))
        pygame.draw.rect(self.canvas, bg_color, game_area_rect)

        game_surface = pygame.Surface((int(self.map_width), int(self.map_height)))
        game_surface.fill(bg_color)

        for tool_name, tool in self.tools.items():
            if tool_name != "delivery_window":
                draw_tool(game_surface, tool, self.tile_size)

        draw_tool(game_surface, self.tools["delivery_window"], self.tile_size)

        for dispenser in self.dispensers:
            draw_dispenser(game_surface, dispenser)

        for essence in self.essences:
            draw_essence(game_surface, essence, self.tile_size)

        draw_player(game_surface, self.player)

        self.canvas.blit(game_surface, (0, self.ui_top_height))

        round_config = self.round_manager.get_current_round()
        if round_config:
            self._draw_timer_bar(
                self.round_time_remaining, round_config["time_limit"], int(self.map_width), self.ui_top_height
            )

            self._draw_requirements(
                self.tools["delivery_window"].required_items,
                int(self.map_width),
                self.ui_top_height + int(self.map_height),
                self.ui_bottom_height,
            )

        # Handle human mode
        if self.render_mode == "human":
            self.window.blit(self.canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

        img = np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

        if img.shape[0] != self.render_size or img.shape[1] != self.render_size:
            img = cv2.resize(img, (self.render_size, self.render_size))

        return img

    def _draw_timer_bar(self, time_remaining: int, time_limit: int, width: int, height: int):
        """Draw timer bar at the top."""
        bar_margin = 10
        bar_height = 30
        bar_width = width - 2 * bar_margin
        bar_x = bar_margin
        bar_y = (height - bar_height) // 2

        pygame.draw.rect(self.canvas, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))

        # Progress
        progress = time_remaining / time_limit if time_limit > 0 else 0
        progress_width = int(bar_width * progress)
        color = (100, 255, 100) if progress > 0.5 else (255, 200, 100) if progress > 0.25 else (255, 100, 100)
        pygame.draw.rect(self.canvas, color, (bar_x, bar_y, progress_width, bar_height))

        # Border
        pygame.draw.rect(self.canvas, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), 3)

        # Time text
        font = pygame.font.SysFont("Arial", 14)
        time_text = f"{time_remaining} / {time_limit}"
        text_surface = font.render(time_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2))
        self.canvas.blit(text_surface, text_rect)

    def _draw_requirements(self, requirements: list[dict], width: int, y_offset: int, height: int):
        """Draw requirements at the bottom - matching floor essence display exactly."""
        margin = 10
        essence_radius = int(0.35 * self.tile_size)
        box_size = essence_radius * 3
        spacing = 10
        start_x = margin
        start_y = y_offset + (height - box_size) // 2

        for i, req in enumerate(requirements):
            box_x = start_x + i * (box_size + spacing)

            # Background color based on completion
            bg_color = (100, 255, 100) if req.get("completed", False) else (80, 80, 85)
            pygame.draw.rect(self.canvas, bg_color, (box_x, start_y, box_size, box_size))
            pygame.draw.rect(self.canvas, (50, 50, 50), (box_x, start_y, box_size, box_size), 2)

            # Draw essence
            essence_types = req["base_essences"]
            enchanted = req.get("enchanted", [False] * len(essence_types))
            refined = req.get("refined", [False] * len(essence_types))
            is_bottled = req.get("bottled", False)

            center = (box_x + box_size // 2, start_y + box_size // 2)

            # Draw bottle
            if is_bottled:
                bottle_rect = pygame.Rect(
                    center[0] - essence_radius * 1.2,
                    center[1] - essence_radius * 1.5,
                    essence_radius * 2.4,
                    essence_radius * 3,
                )
                pygame.draw.rect(self.canvas, (200, 200, 200), bottle_rect, 2)

                neck_rect = pygame.Rect(
                    center[0] - essence_radius * 0.4,
                    center[1] - essence_radius * 1.8,
                    essence_radius * 0.8,
                    essence_radius * 0.5,
                )
                pygame.draw.rect(self.canvas, (200, 200, 200), neck_rect, 2)

            # Draw essence
            if len(essence_types) == 1:
                color = ESSENCE_TYPES[essence_types[0]][1]
                pygame.draw.circle(self.canvas, color, center, essence_radius)

                # Draw patterns if enchanted/refined
                if enchanted[0]:
                    _draw_stripes(self.canvas, center, essence_radius)
                if refined[0]:
                    _draw_dots(self.canvas, center, essence_radius)

            else:
                n_parts = len(essence_types)
                angle_per_part = 360 / n_parts

                for j, etype in enumerate(essence_types):
                    color = ESSENCE_TYPES[etype][1]
                    start_angle = j * angle_per_part  # Same as floor - no rotation
                    end_angle = (j + 1) * angle_per_part

                    # Draw pie slice
                    points = [center]
                    for angle in range(int(start_angle), int(end_angle) + 1, 5):
                        rad = np.deg2rad(angle)  # Same as floor - no -90 offset
                        x = center[0] + essence_radius * np.cos(rad)
                        y = center[1] + essence_radius * np.sin(rad)
                        points.append((int(x), int(y)))
                    points.append(center)
                    pygame.draw.polygon(self.canvas, color, points)

                    # Draw patterns for this slice
                    if enchanted[j]:
                        _draw_stripes_in_slice(self.canvas, center, essence_radius, start_angle, end_angle)
                    if refined[j]:
                        _draw_dots_in_slice(self.canvas, center, essence_radius, start_angle, end_angle)

    def _handle_keyboard_input(self):
        """Handle keyboard input for human control mode."""
        keys = pygame.key.get_pressed()

        # WASD or Arrow keys for movement
        # Using standard screen coordinates: +X right, +Y down
        vx = 0.0
        vy = 0.0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            vx -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            vx += 1.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            vy -= 1.0  # Negative Y to move up (toward y=0)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            vy += 1.0  # Positive Y to move down (toward y=map_height)

        # Normalize diagonal movement
        magnitude = np.sqrt(vx**2 + vy**2)
        if magnitude > 1.0:
            vx /= magnitude
            vy /= magnitude

        self.human_action = np.array([vx, vy], dtype=np.float32)

    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
        self.canvas = None
