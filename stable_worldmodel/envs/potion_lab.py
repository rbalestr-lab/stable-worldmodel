"""
Potion Lab Environment - A physics-based continual learning environment for world models.

This environment tests compositional generalization, catastrophic forgetting, and
long-horizon planning through potion brewing mechanics.
"""

import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces
from typing import Optional, List, Dict, Any

import stable_worldmodel as swm

# Import core game components
from .potion_lab_core import (
    Essence, EssenceState, Player,
    Enchanter, Refiner, Cauldron, Bottler, TrashCan, Dispenser, DeliveryWindow,
    ESSENCE_TYPES,
    PhysicsConfig, setup_physics_space, add_walls,
    CollisionHandler, RoundManager, create_default_layout,
    draw_essence, draw_tool, draw_player, draw_dispenser, draw_ui
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
        render_mode: Optional[str] = "rgb_array",
        resolution: int = 512,
        map_width: int = 16,
        map_height: int = 16,
        tile_size: float = 32.0,
        rounds_config: Optional[List[dict]] = None,
        human_control: bool = False,
    ):
        """
        Initialize the Potion Lab environment.
        
        Args:
            render_mode: "human" or "rgb_array"
            resolution: Render resolution (square image)
            map_width: Width of the map in tiles
            map_height: Height of the map in tiles
            tile_size: Size of each tile in pixels
            rounds_config: Optional list of round configurations
            human_control: If True, enables keyboard control for human play
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.resolution = resolution
        self.map_width_tiles = map_width
        self.map_height_tiles = map_height
        self.tile_size = tile_size
        self.human_control = human_control
        
        # Calculate actual map dimensions in pixels
        self.map_width = map_width * tile_size
        self.map_height = map_height * tile_size
        
        # Action space: continuous 2D velocity control
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space: RGB image
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(resolution, resolution, 3),
                dtype=np.uint8
            )
        })
        
        # Initialize round manager
        if rounds_config is None:
            self.round_manager = RoundManager.create_default_rounds()
        else:
            self.round_manager = RoundManager(rounds_config)
        
        # Pygame and rendering
        self.window = None
        self.clock = None
        self.canvas = None
        
        # Game state
        self.space = None
        self.player = None
        self.essences = []
        self.tools = {}
        self.collision_handler = None
        
        # Timing
        self.step_count = 0
        self.round_time_remaining = 0
        
        # Human control state
        self.human_action = np.array([0.0, 0.0], dtype=np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ):
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        
        # Reset round manager
        self.round_manager.reset()
        
        # Get the first round configuration
        round_config = self.round_manager.get_current_round()
        if round_config is None:
            raise RuntimeError("No rounds configured!")
        
        # Create physics space
        self.space = setup_physics_space()
        
        # Add walls
        add_walls(self.space, self.map_width, self.map_height)
        
        # Create layout (tools, dispensers, player)
        layout = create_default_layout(
            self.space,
            self.map_width,
            self.map_height,
            self.tile_size
        )
        
        self.player = layout['player']
        self.tools = {
            'enchanter': layout['enchanter'],
            'refiner': layout['refiner'],
            'cauldron': layout['cauldron'],
            'bottler': layout['bottler'],
            'trash_can': layout['trash_can'],
            'delivery_window': layout['delivery_window'],
        }
        self.dispensers = layout['dispensers']
        
        # Set up collision handlers
        self.collision_handler = CollisionHandler(self)
        self.collision_handler.setup_handlers(self.space)
        
        # Initialize essences list
        self.essences = []
        
        # Set requirements for delivery window
        requirements = round_config['required_items'].copy()
        for req in requirements:
            req['completed'] = False
        self.tools['delivery_window'].set_requirements(requirements)
        
        # Set round timer
        self.round_time_remaining = round_config['time_limit']
        self.step_count = 0
        
        # Get initial observation
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray):
        """
        Execute one step in the environment.
        
        Args:
            action: [vx, vy] velocity commands in [-1, 1]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # If in human control mode, use keyboard input instead
        if self.human_control:
            action = self.human_action
        
        # Apply action to player
        self.player.apply_action(action)
        
        # Step physics simulation
        self.space.step(PhysicsConfig.TIMESTEP)
        
        # Check for collisions (player with dispensers, essences with tools)
        self.collision_handler.update()
        
        # Update all tools
        for tool in self.tools.values():
            tool.update(PhysicsConfig.TIMESTEP)
        
        # Update dispensers
        for dispenser in self.dispensers:
            dispenser.update(PhysicsConfig.TIMESTEP)
        
        # Check for ejections from tools
        self._check_tool_ejections()
        
        # Update timer
        self.round_time_remaining -= 1
        self.step_count += 1
        
        # Check termination conditions
        terminated = False
        truncated = False
        reward = 0.0
        
        # Check if round is complete
        if self.tools['delivery_window'].all_requirements_met():
            # Round complete! Advance to next round
            reward = 1.0
            self.round_manager.advance_round()
            
            # Check if all rounds are complete
            if self.round_manager.is_complete():
                terminated = True
            else:
                # Load next round
                self._load_next_round()
        
        # Check if time ran out
        elif self.round_time_remaining <= 0:
            # Round failed
            reward = -1.0
            terminated = True
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _load_next_round(self):
        """Load the next round configuration without full reset."""
        round_config = self.round_manager.get_current_round()
        if round_config is None:
            return
        
        # Clear all essences
        for essence in self.essences[:]:
            essence.remove_from_world()
        self.essences.clear()
        
        # Reset tool states
        for tool in self.tools.values():
            if hasattr(tool, 'state'):
                tool.state = type(tool.state)(0)  # Reset to first enum value (EMPTY/WAITING)
                tool.timer = 0
                if hasattr(tool, 'current_essence'):
                    tool.current_essence = None
                if hasattr(tool, 'essence_list'):
                    tool.essence_list = []
        
        # Set new requirements
        requirements = round_config['required_items'].copy()
        for req in requirements:
            req['completed'] = False
        self.tools['delivery_window'].set_requirements(requirements)
        
        # Reset round timer
        self.round_time_remaining = round_config['time_limit']
    
    def _check_tool_ejections(self):
        """Check if any tools are ready to eject processed essences."""
        eject_offset = self.tile_size * 0.8
        
        # Check Enchanter
        if hasattr(self.tools['enchanter'], 'eject_essence'):
            essence_state = self.tools['enchanter'].eject_essence()
            if essence_state is not None:
                pos = (
                    self.tools['enchanter'].position[0] + eject_offset,
                    self.tools['enchanter'].position[1]
                )
                essence = Essence(self.space, pos, essence_state, self.tile_size)
                self.essences.append(essence)
        
        # Check Refiner
        if hasattr(self.tools['refiner'], 'eject_essence'):
            essence_state = self.tools['refiner'].eject_essence()
            if essence_state is not None:
                pos = (
                    self.tools['refiner'].position[0] - eject_offset,
                    self.tools['refiner'].position[1]
                )
                essence = Essence(self.space, pos, essence_state, self.tile_size)
                self.essences.append(essence)
        
        # Check Cauldron
        if hasattr(self.tools['cauldron'], 'eject_essence'):
            essence_state = self.tools['cauldron'].eject_essence()
            if essence_state is not None:
                pos = (
                    self.tools['cauldron'].position[0],
                    self.tools['cauldron'].position[1] + eject_offset
                )
                essence = Essence(self.space, pos, essence_state, self.tile_size)
                self.essences.append(essence)
        
        # Check Bottler
        if hasattr(self.tools['bottler'], 'eject_essence'):
            essence_state = self.tools['bottler'].eject_essence()
            if essence_state is not None:
                pos = (
                    self.tools['bottler'].position[0],
                    self.tools['bottler'].position[1] + eject_offset
                )
                essence = Essence(self.space, pos, essence_state, self.tile_size)
                self.essences.append(essence)
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get the current observation."""
        # Render the frame
        img = self.render()
        
        return {"image": img}
    
    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information."""
        round_config = self.round_manager.get_current_round()
        
        return {
            "round_index": self.round_manager.current_round_index,
            "time_remaining": self.round_time_remaining,
            "time_limit": round_config['time_limit'] if round_config else 0,
            "requirements_met": self.tools['delivery_window'].all_requirements_met(),
            "num_essences": len(self.essences),
            "player_pos": np.array([self.player.body.position.x, self.player.body.position.y]),
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        # UI padding
        ui_top_height = 50
        ui_bottom_height = 80
        total_height = int(self.map_height + ui_top_height + ui_bottom_height)
        
        # Initialize pygame if needed
        if self.canvas is None:
            pygame.init()
            self.canvas = pygame.Surface((int(self.map_width), total_height))
            
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((int(self.map_width), total_height))
                pygame.display.set_caption("Potion Lab")
                self.clock = pygame.time.Clock()
        
        # Clear canvas
        self.canvas.fill((30, 30, 35))  # Dark background for UI areas
        
        # Fill game area with different color
        game_area_rect = pygame.Rect(0, ui_top_height, int(self.map_width), int(self.map_height))
        pygame.draw.rect(self.canvas, (40, 40, 45), game_area_rect)
        
        # Create temporary surface for game area (offset for rendering)
        game_surface = pygame.Surface((int(self.map_width), int(self.map_height)))
        game_surface.fill((40, 40, 45))
        
        # Draw game objects on game surface
        for tool_name, tool in self.tools.items():
            if tool_name != 'delivery_window':
                draw_tool(game_surface, tool, self.tile_size)
        
        draw_tool(game_surface, self.tools['delivery_window'], self.tile_size)
        
        for dispenser in self.dispensers:
            draw_dispenser(game_surface, dispenser)
        
        for essence in self.essences:
            draw_essence(game_surface, essence, self.tile_size)
        
        draw_player(game_surface, self.player)
        
        # Blit game surface to main canvas at offset
        self.canvas.blit(game_surface, (0, ui_top_height))
        
        # Draw UI elements outside game area
        round_config = self.round_manager.get_current_round()
        if round_config:
            # Draw timer at top
            self._draw_timer_bar(
                self.round_time_remaining,
                round_config['time_limit'],
                int(self.map_width),
                ui_top_height
            )
            
            # Draw requirements at bottom
            self._draw_requirements(
                self.tools['delivery_window'].required_items,
                int(self.map_width),
                ui_top_height + int(self.map_height),
                ui_bottom_height
            )
        
        # Handle human mode
        if self.render_mode == "human":
            # Handle keyboard input for human control
            if self.human_control:
                self._handle_keyboard_input()
            
            # Copy to display window
            self.window.blit(self.canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
        
        # Convert to numpy array and resize
        img = np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))
        
        # Resize to target resolution
        if img.shape[0] != self.resolution or img.shape[1] != self.resolution:
            import cv2
            img = cv2.resize(img, (self.resolution, self.resolution))
        
        return img
    
    def _draw_timer_bar(self, time_remaining: int, time_limit: int, width: int, height: int):
        """Draw timer bar at the top."""
        bar_margin = 10
        bar_height = 30
        bar_width = width - 2 * bar_margin
        bar_x = bar_margin
        bar_y = (height - bar_height) // 2
        
        # Background
        pygame.draw.rect(self.canvas, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
        
        # Progress
        progress = time_remaining / time_limit if time_limit > 0 else 0
        progress_width = int(bar_width * progress)
        color = (100, 255, 100) if progress > 0.5 else (255, 200, 100) if progress > 0.25 else (255, 100, 100)
        pygame.draw.rect(self.canvas, color, (bar_x, bar_y, progress_width, bar_height))
        
        # Border
        pygame.draw.rect(self.canvas, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), 3)
        
        # Time text
        font = pygame.font.SysFont('Arial', 14)
        time_text = f"{time_remaining} / {time_limit}"
        text_surface = font.render(time_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2))
        self.canvas.blit(text_surface, text_rect)
    
    def _draw_requirements(self, requirements: List[dict], width: int, y_offset: int, height: int):
        """Draw requirements at the bottom."""
        margin = 10
        box_size = 50
        spacing = 10
        start_x = margin
        start_y = y_offset + (height - box_size) // 2
        
        font_small = pygame.font.SysFont('Arial', 12)
        
        for i, req in enumerate(requirements):
            box_x = start_x + i * (box_size + spacing)
            
            # Background color based on completion
            from .potion_lab_core import ESSENCE_TYPES
            bg_color = (100, 255, 100) if req.get('completed', False) else (80, 80, 85)
            pygame.draw.rect(self.canvas, bg_color, (box_x, start_y, box_size, box_size))
            pygame.draw.rect(self.canvas, (50, 50, 50), (box_x, start_y, box_size, box_size), 2)
            
            # Draw simplified essence representation
            essence_types = req['base_essences']
            if len(essence_types) == 1:
                color = ESSENCE_TYPES[essence_types[0]][1]
                center = (box_x + box_size // 2, start_y + box_size // 2)
                pygame.draw.circle(self.canvas, color, center, box_size // 3)
            else:
                # Multiple essences - draw as segments
                n_parts = len(essence_types)
                segment_width = box_size // n_parts
                for j, etype in enumerate(essence_types):
                    color = ESSENCE_TYPES[etype][1]
                    seg_rect = pygame.Rect(box_x + j * segment_width, start_y, segment_width, box_size)
                    pygame.draw.rect(self.canvas, color, seg_rect)
            
            # Checkmark if completed
            if req.get('completed', False):
                check_text = font_small.render('✓', True, (255, 255, 255))
                self.canvas.blit(check_text, (box_x + 5, start_y + 5))
    
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


# ============================================================================
# Human Play Function
# ============================================================================

def play_potion_lab():
    """
    Launch the Potion Lab environment in human-playable mode.
    
    Controls:
    - WASD or Arrow Keys: Move the player
    - ESC: Quit
    
    Usage:
        >>> from stable_worldmodel.envs.potion_lab import play_potion_lab
        >>> play_potion_lab()
    """
    print("=" * 60)
    print("POTION LAB - Human Play Mode")
    print("=" * 60)
    print("\nControls:")
    print("  WASD or Arrow Keys: Move the player")
    print("  ESC: Quit")
    print("\nObjective:")
    print("  - Collect essences from dispensers (colored squares at top)")
    print("  - Process them through tools (Enchanter, Refiner, Cauldron)")
    print("  - Bottle them if required")
    print("  - Deliver to the delivery window at the bottom")
    print("\nPress any key in the game window to start...")
    print("=" * 60)
    
    env = PotionLab(
        render_mode="human",
        resolution=512,
        human_control=True
    )
    
    obs, info = env.reset()
    
    running = True
    while running:
        # Step the environment (human_control=True means keyboard input is used)
        obs, reward, terminated, truncated, info = env.step(env.human_action)
        
        # Check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Reset if episode ends
        if terminated or truncated:
            if terminated:
                if reward > 0:
                    print(f"\nRound {info['round_index']} completed!")
                else:
                    print(f"\nRound {info['round_index']} failed (time ran out)")
            
            # Wait a bit before reset
            pygame.time.wait(2000)
            
            # Check if all rounds complete
            if env.round_manager.is_complete():
                print("\n" + "=" * 60)
                print("ALL ROUNDS COMPLETE! Well done!")
                print("=" * 60)
                running = False
            else:
                obs, info = env.reset()
    
    env.close()
    print("\nThanks for playing!")


if __name__ == "__main__":
    play_potion_lab()

