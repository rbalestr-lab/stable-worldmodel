if __name__ == "__main__":
    import numpy as np
    import pygame

    from stable_worldmodel.envs.potion_lab import PotionLab

    ######################
    ##  World Creation  ##
    ######################

    # Create the PotionLab environment directly
    env = PotionLab(
        render_mode="human",
        resolution=512,
        render_action=False,
    )

    print("Potion Lab - Move mouse to control player, ESC to quit")
    print("Collect essences, process them through tools, and deliver potions!")

    #######################
    ##  Play Loop       ##
    #######################

    obs, info = env.reset(seed=42)
    running = True

    # Initialize pygame clock (pygame.init() is called in env.render())
    clock = None

    # Initial render to set up pygame window
    env.render()
    if clock is None:
        clock = pygame.time.Clock()

    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Get mouse position and player position
        # Mouse position is relative to the pygame window
        mouse_x, mouse_y = pygame.mouse.get_pos()
        player_pos = info["player_pos"]  # [x, y] in game coordinates

        # Convert mouse position from screen coordinates to game coordinates
        # Game area starts at y = ui_top_height (50 pixels from top)
        # Mouse y needs to be adjusted: mouse_y - ui_top_height
        ui_top_height = env.ui_top_height
        game_mouse_x = np.clip(mouse_x, 0, env.map_width)
        game_mouse_y = np.clip(mouse_y - ui_top_height, 0, env.map_height)

        # Calculate direction vector from player to mouse
        dx = game_mouse_x - player_pos[0]
        dy = game_mouse_y - player_pos[1]

        # Calculate distance and normalize to [-1, 1] range
        distance = np.sqrt(dx**2 + dy**2)

        # Only move if mouse is far enough from player (dead zone)
        dead_zone = 5.0  # pixels
        if distance > dead_zone:
            # Normalize direction vector
            vx = dx / distance
            vy = dy / distance

            vx = np.clip(vx, -1.0, 1.0)
            vy = np.clip(vy, -1.0, 1.0)
        else:
            # Stop moving if mouse is too close
            vx = 0.0
            vy = 0.0

        action = np.array([vx, vy], dtype=np.float32)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render (handled automatically in human mode)
        env.render()

        # Check if episode ended
        if terminated or truncated:
            if terminated:
                if reward > 0:
                    print(f"Round {info['round_index']} completed! Reward: {reward:.2f}")
                else:
                    print(f"Round {info['round_index']} failed! Time ran out.")
            else:
                print(f"Episode truncated at step {info.get('time_remaining', 0)}")

            # Reset for next episode
            obs, info = env.reset(seed=42)
            print(f"Starting round {info['round_index'] + 1}")

        # Control frame rate
        if clock is not None:
            clock.tick(60)

    env.close()
    print("Thanks for playing!")
