import gymnasium as gym
import pymunk

from .pusht import PushT


class PushTPhysics(PushT):
    def __init__(
        self,
        block_mass: float = 1.0,
        friction: float = 1.0,
        damping: float = 0.0,
        action_force_scale: float = 1.0,
        **kwargs
    ):
        # Validate physics parameters
        if block_mass <= 0:
            raise ValueError(f"block_mass must be positive, got {block_mass}")
        if friction < 0:
            raise ValueError(f"friction must be non-negative, got {friction}")
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}")
        if action_force_scale <= 0:
            raise ValueError(f"action_force_scale must be positive, got {action_force_scale}")

        self.physics_config = {
            'block_mass': block_mass,
            'friction': friction,
            'damping': damping,
            'action_force_scale': action_force_scale,
        }

        if 'damping' not in kwargs:
            kwargs['damping'] = damping

        super().__init__(**kwargs)
    
    def add_circle(self, position, angle=0, scale=1, color="RoyalBlue"):
        body = super().add_circle(position, angle, scale, color)

        # Apply custom friction
        for shape in body.shapes:
            shape.friction = self.physics_config['friction']

        return body
    
    def add_tee(self, position, angle, scale=30, color="LightSlateGray", mask=pymunk.ShapeFilter.ALL_MASKS()):
        body = super().add_tee(position, angle, scale, color, mask)
        return self._apply_block_physics(body)

    def add_small_tee(self, position, angle, scale=30, color="LightSlateGray", mask=pymunk.ShapeFilter.ALL_MASKS()):
        body = super().add_small_tee(position, angle, scale, color, mask)
        return self._apply_block_physics(body)

    def add_plus(self, position, angle, scale=30, color="LightSlateGray", mask=pymunk.ShapeFilter.ALL_MASKS()):
        body = super().add_plus(position, angle, scale, color, mask)
        return self._apply_block_physics(body)

    def add_L(self, position, angle, scale=30, color="LightSlateGray", mask=pymunk.ShapeFilter.ALL_MASKS()):
        body = super().add_L(position, angle, scale, color, mask)
        return self._apply_block_physics(body)

    def add_Z(self, position, angle, scale=30, color="LightSlateGray", mask=pymunk.ShapeFilter.ALL_MASKS()):
        body = super().add_Z(position, angle, scale, color, mask)
        return self._apply_block_physics(body)

    def add_square(self, position, angle, scale=30, color="LightSlateGray", mask=pymunk.ShapeFilter.ALL_MASKS()):
        body = super().add_square(position, angle, scale, color, mask)
        return self._apply_block_physics(body)

    def add_I(self, position, angle, scale=30, color="LightSlateGray", mask=pymunk.ShapeFilter.ALL_MASKS()):
        body = super().add_I(position, angle, scale, color, mask)
        return self._apply_block_physics(body)
    
    def _apply_block_physics(self, body):
        # Apply mass multiplier
        if self.physics_config['block_mass'] != 1.0:
            body.mass *= self.physics_config['block_mass']
            body.moment *= self.physics_config['block_mass']

        # Apply friction
        for shape in body.shapes:
            shape.friction = self.physics_config['friction']

        return body
    
    def step(self, action):
        if self.physics_config['action_force_scale'] != 1.0:
            action = action * self.physics_config['action_force_scale']

        return super().step(action)


# Register with Gymnasium
gym.register(
    id='PushTPhysics-v1',
    entry_point='stable_worldmodel.envs:PushTPhysics',
    max_episode_steps=200,
)

