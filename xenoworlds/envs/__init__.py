from gymnasium.envs.registration import register

register(
    id="xenoworlds/ImagePositioning",
    entry_point="xenowrolds.envs.image_positioning:ImagePositioning",
)
