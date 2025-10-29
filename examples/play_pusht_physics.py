#!/usr/bin/env python3
"""Interactive PushTPhysics Demo - Play and adjust physics in real-time!

Controls:
    Mouse: Move the agent (click and drag or just move)
    
Physics Controls:
    1/2: Decrease/Increase block mass
    3/4: Decrease/Increase friction
    5/6: Decrease/Increase damping
    7/8: Decrease/Increase action force scale
    
Other Controls:
    R: Reset environment
    SPACE: Toggle auto-reset on success
    P: Print current physics config
    ESC/Q: Quit
    
The display shows:
    - Current physics parameters (top-left)
    - Goal position (light green T)
    - Current block (gray T)
    - Agent (blue circle)
"""

import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import pygame

# Add stable_worldmodel to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))
import stable_worldmodel as swm


class InteractivePushTPhysics:
    """Interactive demo for PushTPhysics with real-time physics adjustment."""
    
    def __init__(self, window_size=512, render_size=512):
        """Initialize the interactive demo.
        
        Args:
            window_size: Size of the pygame window
            render_size: Size of the rendered environment
        """
        self.window_size = window_size
        self.render_size = render_size
        
        # Initial physics parameters
        self.physics_params = {
            'block_mass': 1.0,
            'friction': 1.0,
            'damping': 0.0,
            'action_force_scale': 1.0,
        }
        
        # Parameter adjustment step sizes
        self.param_steps = {
            'block_mass': 0.1,
            'friction': 0.1,
            'damping': 0.05,
            'action_force_scale': 0.1,
        }
        
        # Parameter limits
        self.param_limits = {
            'block_mass': (0.1, 5.0),
            'friction': (0.0, 3.0),
            'damping': (0.0, 0.8),
            'action_force_scale': (0.1, 3.0),
        }
        
        # Create initial environment
        self.env = None
        self.recreate_env()
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("PushTPhysics Interactive Demo")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # State
        self.running = True
        self.mouse_pos = None
        self.auto_reset = False
        self.total_reward = 0
        self.steps = 0
        self.success_count = 0
        
    def recreate_env(self):
        """Recreate environment with current physics parameters."""
        if self.env is not None:
            self.env.close()
        
        self.env = gym.make(
            "swm/PushTPhysics-v1",
            render_mode="rgb_array",
            resolution=self.render_size,
            **self.physics_params
        )
        self.obs, self.info = self.env.reset(seed=None)
        self.total_reward = 0
        self.steps = 0
        
    def adjust_parameter(self, param_name, direction):
        """Adjust a physics parameter and recreate environment.
        
        Args:
            param_name: Name of parameter to adjust
            direction: +1 to increase, -1 to decrease
        """
        current = self.physics_params[param_name]
        step = self.param_steps[param_name]
        new_value = current + (direction * step)
        
        # Clamp to limits
        min_val, max_val = self.param_limits[param_name]
        new_value = np.clip(new_value, min_val, max_val)
        
        if new_value != current:
            self.physics_params[param_name] = new_value
            print(f"\n{param_name}: {current:.2f} → {new_value:.2f}")
            self.recreate_env()
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                # Quit
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                
                # Reset
                elif event.key == pygame.K_r:
                    print("\nResetting environment...")
                    self.obs, self.info = self.env.reset()
                    self.total_reward = 0
                    self.steps = 0
                
                # Toggle auto-reset
                elif event.key == pygame.K_SPACE:
                    self.auto_reset = not self.auto_reset
                    print(f"\nAuto-reset on success: {self.auto_reset}")
                
                # Print config
                elif event.key == pygame.K_p:
                    self.print_config()
                
                # Adjust block mass
                elif event.key == pygame.K_1:
                    self.adjust_parameter('block_mass', -1)
                elif event.key == pygame.K_2:
                    self.adjust_parameter('block_mass', +1)
                
                # Adjust friction
                elif event.key == pygame.K_3:
                    self.adjust_parameter('friction', -1)
                elif event.key == pygame.K_4:
                    self.adjust_parameter('friction', +1)
                
                # Adjust damping
                elif event.key == pygame.K_5:
                    self.adjust_parameter('damping', -1)
                elif event.key == pygame.K_6:
                    self.adjust_parameter('damping', +1)
                
                # Adjust action force scale
                elif event.key == pygame.K_7:
                    self.adjust_parameter('action_force_scale', -1)
                elif event.key == pygame.K_8:
                    self.adjust_parameter('action_force_scale', +1)
    
    def get_mouse_action(self):
        """Get action from mouse position.
        
        Returns:
            action: 2D action in [-1, 1] range
        """
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Convert mouse position to environment coordinates
        # Pygame coords: (0,0) top-left, (512, 512) bottom-right
        # Env coords: (0,0) top-left, (512, 512) bottom-right (same!)
        scale = 512.0 / self.window_size
        env_x = mouse_x * scale
        env_y = mouse_y * scale
        
        # Get agent position from observation
        agent_pos = self.obs['state'][:2]
        
        # Action is direction to mouse (normalized to [-1, 1])
        target = np.array([env_x, env_y])
        direction = target - agent_pos
        
        # Normalize to action space
        action = direction / 512.0 * 2  # Scale to roughly [-1, 1]
        action = np.clip(action, -1, 1).astype(np.float32)
        
        return action
    
    def render_ui(self, frame):
        """Render UI overlay on the frame.
        
        Args:
            frame: RGB image from environment
            
        Returns:
            frame with UI overlay
        """
        # Convert to pygame surface
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        frame_surface = pygame.transform.scale(frame_surface, (self.window_size, self.window_size))
        
        self.screen.blit(frame_surface, (0, 0))
        
        # Draw semi-transparent overlay for text background
        overlay = pygame.Surface((300, 200))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (10, 10))
        
        # Draw physics parameters
        y_offset = 20
        title = self.font.render("Physics Parameters:", True, (255, 255, 100))
        self.screen.blit(title, (20, y_offset))
        y_offset += 30
        
        params_display = [
            f"1/2 Block Mass: {self.physics_params['block_mass']:.2f}",
            f"3/4 Friction: {self.physics_params['friction']:.2f}",
            f"5/6 Damping: {self.physics_params['damping']:.2f}",
            f"7/8 Force Scale: {self.physics_params['action_force_scale']:.2f}",
        ]
        
        for text in params_display:
            surface = self.small_font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (20, y_offset))
            y_offset += 25
        
        # Draw status info
        y_offset += 10
        status = [
            f"Steps: {self.steps}",
            f"Reward: {self.total_reward:.1f}",
            f"Successes: {self.success_count}",
        ]
        
        for text in status:
            surface = self.small_font.render(text, True, (100, 255, 100))
            self.screen.blit(surface, (20, y_offset))
            y_offset += 20
        
        # Draw controls reminder at bottom
        controls_y = self.window_size - 100
        overlay_bottom = pygame.Surface((self.window_size - 20, 90))
        overlay_bottom.set_alpha(200)
        overlay_bottom.fill((0, 0, 0))
        self.screen.blit(overlay_bottom, (10, controls_y))
        
        controls = [
            "Mouse: Move agent | 1-8: Adjust physics",
            "R: Reset | Space: Auto-reset | P: Print | Q: Quit",
        ]
        
        for i, text in enumerate(controls):
            surface = self.small_font.render(text, True, (200, 200, 255))
            self.screen.blit(surface, (20, controls_y + 10 + i * 25))
        
        pygame.display.flip()
    
    def print_config(self):
        """Print current configuration to console."""
        print("\n" + "=" * 60)
        print("Current Physics Configuration:")
        print("=" * 60)
        for param, value in self.physics_params.items():
            print(f"  {param:20s}: {value:.2f}")
        print(f"\n  Steps taken: {self.steps}")
        print(f"  Total reward: {self.total_reward:.2f}")
        print(f"  Successes: {self.success_count}")
        print("=" * 60 + "\n")
    
    def step(self):
        """Execute one step of the demo."""
        # Get action from mouse
        action = self.get_mouse_action()
        
        # Step environment
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.total_reward += reward
        self.steps += 1
        
        # Check for success
        if terminated:
            self.success_count += 1
            print(f"\n🎉 Success! Total steps: {self.steps}, Reward: {self.total_reward:.1f}")
            
            if self.auto_reset:
                print("Auto-resetting...")
                self.obs, self.info = self.env.reset()
                self.total_reward = 0
                self.steps = 0
        
        # Render
        frame = self.env.render()
        self.render_ui(frame)
    
    def run(self):
        """Main loop."""
        print("\n" + "=" * 70)
        print("🎮 PushTPhysics Interactive Demo")
        print("=" * 70)
        print("\nControls:")
        print("  Mouse:        Move the agent")
        print("  1/2:          Decrease/Increase block mass")
        print("  3/4:          Decrease/Increase friction")
        print("  5/6:          Decrease/Increase damping")
        print("  7/8:          Decrease/Increase action force scale")
        print("  R:            Reset environment")
        print("  SPACE:        Toggle auto-reset on success")
        print("  P:            Print current physics config")
        print("  ESC/Q:        Quit")
        print("=" * 70)
        print("\nStarting demo...\n")
        
        while self.running:
            self.handle_events()
            self.step()
            self.clock.tick(30)  # 30 FPS
        
        print("\nCleaning up...")
        self.env.close()
        pygame.quit()
        print("Demo closed. Thanks for playing! 👋\n")


def main():
    """Run the interactive demo."""
    try:
        demo = InteractivePushTPhysics(window_size=512, render_size=512)
        demo.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        pygame.quit()
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()


if __name__ == "__main__":
    main()

