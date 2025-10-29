#!/usr/bin/env python3
"""Interactive PushTPhysics Demo with GUI Sliders

A more user-friendly interface with sliders to adjust physics parameters.
Uses pygame for the environment display and simple button/slider UI.

Controls:
    Mouse: Move the agent (in game area)
    Click sliders: Adjust physics parameters
    Reset button: Reset environment
    New Physics button: Apply physics changes
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame

# Add stable_worldmodel to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))
import stable_worldmodel as swm


class Slider:
    """Simple slider widget for pygame."""
    
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
        
    def handle_event(self, event):
        """Handle mouse events for the slider."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.update_value(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.update_value(event.pos[0])
    
    def update_value(self, mouse_x):
        """Update slider value based on mouse position."""
        relative_x = mouse_x - self.rect.x
        relative_x = np.clip(relative_x, 0, self.rect.width)
        ratio = relative_x / self.rect.width
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
    
    def draw(self, screen, font):
        """Draw the slider."""
        # Background
        pygame.draw.rect(screen, (80, 80, 80), self.rect)
        
        # Filled portion
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        filled_width = int(self.rect.width * ratio)
        filled_rect = pygame.Rect(self.rect.x, self.rect.y, filled_width, self.rect.height)
        pygame.draw.rect(screen, (100, 150, 255), filled_rect)
        
        # Border
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)
        
        # Label and value
        label_surf = font.render(f"{self.label}: {self.value:.2f}", True, (255, 255, 255))
        screen.blit(label_surf, (self.rect.x, self.rect.y - 25))


class Button:
    """Simple button widget for pygame."""
    
    def __init__(self, x, y, width, height, text, color=(100, 200, 100)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = tuple(min(c + 30, 255) for c in color)
        self.is_hovered = False
    
    def handle_event(self, event):
        """Handle mouse events for the button."""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False
    
    def draw(self, screen, font):
        """Draw the button."""
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)
        
        text_surf = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)


class PushTPhysicsGUI:
    """Interactive PushTPhysics demo with GUI controls."""
    
    def __init__(self):
        """Initialize the GUI demo."""
        # Window layout
        self.game_size = 512
        self.panel_width = 300
        self.window_width = self.game_size + self.panel_width
        self.window_height = self.game_size
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("PushTPhysics - Interactive GUI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 28)
        
        # Physics parameters
        self.physics_params = {
            'block_mass': 1.0,
            'friction': 1.0,
            'damping': 0.0,
            'action_force_scale': 1.0,
        }
        
        # Create sliders
        slider_x = self.game_size + 20
        slider_width = self.panel_width - 40
        slider_height = 25
        
        self.sliders = {
            'block_mass': Slider(slider_x, 80, slider_width, slider_height, 
                                0.1, 5.0, 1.0, "Block Mass"),
            'friction': Slider(slider_x, 150, slider_width, slider_height,
                             0.0, 3.0, 1.0, "Friction"),
            'damping': Slider(slider_x, 220, slider_width, slider_height,
                            0.0, 0.8, 0.0, "Damping"),
            'action_force_scale': Slider(slider_x, 290, slider_width, slider_height,
                                        0.1, 3.0, 1.0, "Force Scale"),
        }
        
        # Create buttons
        button_width = (self.panel_width - 50) // 2
        button_height = 40
        button_y = 380
        
        self.apply_button = Button(
            slider_x, button_y, button_width, button_height,
            "Apply Physics", (100, 150, 255)
        )
        self.reset_button = Button(
            slider_x + button_width + 10, button_y, button_width, button_height,
            "Reset Env", (255, 150, 100)
        )
        
        # Create environment
        self.env = None
        self.recreate_env()
        
        # State
        self.running = True
        self.steps = 0
        self.total_reward = 0
        self.success_count = 0
        self.in_game_area = False
    
    def recreate_env(self):
        """Recreate environment with current physics parameters."""
        if self.env is not None:
            self.env.close()
        
        self.env = gym.make(
            "swm/PushTPhysics-v1",
            render_mode="rgb_array",
            resolution=self.game_size,
            **self.physics_params
        )
        self.obs, self.info = self.env.reset()
        self.steps = 0
        self.total_reward = 0
    
    def apply_physics_from_sliders(self):
        """Update physics parameters from slider values."""
        changed = False
        for key, slider in self.sliders.items():
            if abs(self.physics_params[key] - slider.value) > 0.01:
                changed = True
            self.physics_params[key] = slider.value
        
        if changed:
            print("\nApplying new physics:")
            for key, value in self.physics_params.items():
                print(f"  {key}: {value:.2f}")
            self.recreate_env()
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_r:
                    self.obs, self.info = self.env.reset()
                    self.steps = 0
                    self.total_reward = 0
            
            elif event.type == pygame.MOUSEMOTION:
                # Check if mouse is in game area
                self.in_game_area = event.pos[0] < self.game_size
            
            # Handle slider events
            for slider in self.sliders.values():
                slider.handle_event(event)
            
            # Handle button events
            if self.apply_button.handle_event(event):
                self.apply_physics_from_sliders()
            
            if self.reset_button.handle_event(event):
                self.obs, self.info = self.env.reset()
                self.steps = 0
                self.total_reward = 0
                print("\nEnvironment reset")
    
    def get_mouse_action(self):
        """Get action from mouse position."""
        if not self.in_game_area:
            # If mouse is over UI panel, don't move
            return np.array([0.0, 0.0], dtype=np.float32)
        
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Convert to environment coordinates
        env_x = mouse_x
        env_y = mouse_y
        
        # Get agent position
        agent_pos = self.obs['state'][:2]
        
        # Calculate action
        target = np.array([env_x, env_y])
        direction = target - agent_pos
        action = direction / 512.0 * 2
        action = np.clip(action, -1, 1).astype(np.float32)
        
        return action
    
    def draw_ui_panel(self):
        """Draw the control panel."""
        # Panel background
        panel_rect = pygame.Rect(self.game_size, 0, self.panel_width, self.window_height)
        pygame.draw.rect(self.screen, (30, 30, 40), panel_rect)
        
        # Title
        title = self.title_font.render("Physics Controls", True, (255, 255, 100))
        self.screen.blit(title, (self.game_size + 20, 20))
        
        # Draw sliders
        for slider in self.sliders.values():
            slider.draw(self.screen, self.font)
        
        # Draw buttons
        self.apply_button.draw(self.screen, self.font)
        self.reset_button.draw(self.screen, self.font)
        
        # Draw status info
        status_y = 450
        status_texts = [
            f"Steps: {self.steps}",
            f"Reward: {self.total_reward:.1f}",
            f"Successes: {self.success_count}",
        ]
        
        for i, text in enumerate(status_texts):
            surf = self.font.render(text, True, (150, 255, 150))
            self.screen.blit(surf, (self.game_size + 20, status_y + i * 25))
        
        # Instructions
        instructions_y = self.window_height - 60
        instructions = [
            "Move mouse in game area to play",
            "Adjust sliders, then click Apply",
        ]
        
        for i, text in enumerate(instructions):
            surf = self.font.render(text, True, (200, 200, 255))
            self.screen.blit(surf, (self.game_size + 20, instructions_y + i * 20))
    
    def step(self):
        """Execute one step."""
        # Get action
        action = self.get_mouse_action()
        
        # Step environment
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.total_reward += reward
        self.steps += 1
        
        # Check for success
        if terminated:
            self.success_count += 1
            print(f"\n🎉 Success! Steps: {self.steps}, Reward: {self.total_reward:.1f}")
            self.obs, self.info = self.env.reset()
            self.steps = 0
            self.total_reward = 0
        
        # Render environment
        frame = self.env.render()
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(frame_surface, (0, 0))
        
        # Draw UI panel
        self.draw_ui_panel()
        
        pygame.display.flip()
    
    def run(self):
        """Main loop."""
        print("\n" + "=" * 70)
        print("🎮 PushTPhysics Interactive GUI Demo")
        print("=" * 70)
        print("\nHow to use:")
        print("  1. Move your mouse in the game area to control the agent")
        print("  2. Adjust the physics sliders in the right panel")
        print("  3. Click 'Apply Physics' to update the environment")
        print("  4. Click 'Reset Env' to restart the current task")
        print("  5. Press ESC or Q to quit")
        print("=" * 70 + "\n")
        
        while self.running:
            self.handle_events()
            self.step()
            self.clock.tick(30)  # 30 FPS
        
        print("\nCleaning up...")
        self.env.close()
        pygame.quit()
        print("Demo closed. Thanks for playing! 👋\n")


def main():
    """Run the GUI demo."""
    try:
        demo = PushTPhysicsGUI()
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

