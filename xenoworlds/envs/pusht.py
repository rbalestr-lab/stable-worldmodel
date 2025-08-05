# ----------------------------------------------------------------------------
# pymunk
# Copyright (c) 2007-2016 Victor Blomqvist
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

# code from: https://github.com/huggingface/gym-pusht/tree/main/gym_pusht/envs

import collections
import os
import warnings
from typing import Sequence, Tuple

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gymnasium import spaces
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d

positive_y_is_up: bool = False


with warnings.catch_warnings():
    # Filter out DeprecationWarnings raised from pkg_resources
    warnings.filterwarnings(
        "ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning
    )
    import pygame


RENDER_MODES = ["rgb_array"]
if os.environ.get("MUJOCO_GL") != "egl":
    RENDER_MODES.append("human")


def pymunk_to_shapely(body, shapes):
    geoms = []
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class PushT(gym.Env):
    """
    ## Description

    PushT environment.

    The goal of the agent is to push the block to the goal zone. The agent is a circle and the block is a tee shape.

    ## Action Space

    The action space is continuous and consists of two values: [x, y]. The values are in the range [0, 512] and
    represent the target position of the agent.

    ## Observation Space

    If `obs_type` is set to `state`, the observation space is a 5-dimensional vector representing the state of the
    environment: [agent_x, agent_y, block_x, block_y, block_angle]. The values are in the range [0, 512] for the agent
    and block positions and [0, 2*pi] for the block angle.

    If `obs_type` is set to `environment_state_agent_pos` the observation space is a dictionary with:
    - `environment_state`: 16-dimensional vector representing the keypoint locations of the T (in [x0, y0, x1, y1, ...]
        format). The values are in the range [0, 512]. See `get_keypoints` for a diagram showing the location of the
        keypoint indices.
    - `agent_pos`: A 2-dimensional vector representing the position of the robot end-effector.

    If `obs_type` is set to `pixels`, the observation space is a 96x96 RGB image of the environment.

    ## Rewards

    The reward is the coverage of the block in the goal zone. The reward is 1.0 if the block is fully in the goal zone.

    ## Success Criteria

    The environment is considered solved if the block is at least 95% in the goal zone.

    ## Starting State

    The agent starts at a random position and the block starts at a random position and angle.

    ## Episode Termination

    The episode terminates when the block is at least 95% in the goal zone.

    ## Arguments

    ```python
    >>> import gymnasium as gym
    >>> import gym_pusht
    >>> env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<PushT<gym_pusht/PushT-v0>>>>>
    ```

    * `obs_type`: (str) The observation type. Can be either `state`, `keypoints`, `pixels` or `pixels_agent_pos`.
      Default is `state`.

    * `block_cog`: (tuple) The center of gravity of the block if different from the center of mass. Default is `None`.

    * `damping`: (float) The damping factor of the environment if different from 0. Default is `None`.

    * `observation_width`: (int) The width of the observed image. Default is `96`.

    * `observation_height`: (int) The height of the observed image. Default is `96`.

    * `visualization_width`: (int) The width of the visualized image. Default is `680`.

    * `visualization_height`: (int) The height of the visualized image. Default is `680`.

    ## Reset Arguments

    Passing the option `options["reset_to_state"]` will reset the environment to a specific state.

    > [!WARNING]
    > For legacy compatibility, the inner fonctionning has been preserved, and the state set is not the same as the
    > the one passed in the argument.

    ```python
    >>> import gymnasium as gym
    >>> import gym_pusht
    >>> env = gym.make("gym_pusht/PushT-v0")
    >>> state, _ = env.reset(options={"reset_to_state": [0.0, 10.0, 20.0, 30.0, 1.0]})
    >>> state
    array([ 0.      , 10.      , 57.866196, 50.686398,  1.      ],
          dtype=float32)
    ```

    ## Version History

    * v0: Original version

    ## References

    * TODO:
    """

    metadata = {"render_modes": RENDER_MODES, "render_fps": 10}

    def __init__(
        self,
        obs_type="state",
        render_mode="rgb_array",
        block_cog=None,
        damping=None,
        resolution=224,
        visualization_width=680,
        visualization_height=680,
    ):
        super().__init__()
        # Observations
        self.obs_type = obs_type

        # Rendering
        self.render_mode = render_mode
        self.observation_width = resolution
        self.observation_height = resolution
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        # Initialize spaces
        self._initialize_observation_space()
        self.action_space = spaces.Box(low=0, high=512, shape=(2,), dtype=np.float32)

        # Physics
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["render_fps"]
        self.dt = 0.01
        self.block_cog = block_cog
        self.damping = damping

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None

        self.teleop = None
        self._last_action = None

        self.success_threshold = 0.95  # 95% coverage

    def _initialize_observation_space(self):
        if self.obs_type == "state":
            # [agent_x, agent_y, block_x, block_y, block_angle]
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0]),
                high=np.array([512, 512, 512, 512, 2 * np.pi]),
                dtype=np.float64,
            )
        elif self.obs_type == "environment_state_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "environment_state": spaces.Box(
                        low=np.zeros(16),
                        high=np.full((16,), 512),
                        dtype=np.float64,
                    ),
                    "agent_pos": spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([512, 512]),
                        dtype=np.float64,
                    ),
                },
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict({
                "pixels": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.observation_height, self.observation_width, 3),
                    dtype=np.uint8,
                ),
                "agent_pos": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([512, 512]),
                    dtype=np.float64,
                ),
            })
        else:
            raise ValueError(
                f"Unknown obs_type {self.obs_type}. Must be one of [pixels, state, environment_state_agent_pos, "
                "pixels_agent_pos]"
            )

    def _get_coverage(self):
        goal_body = self.get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        return intersection_area / goal_area

    def step(self, action):
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        self._last_action = action
        for _ in range(n_steps):
            # Step PD control
            # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
            acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                Vec2d(0, 0) - self.agent.velocity
            )
            self.agent.velocity += acceleration * self.dt

            # Step physics
            self.space.step(self.dt)

        # Compute reward
        coverage = self._get_coverage()
        reward = np.clip(coverage / self.success_threshold, 0.0, 1.0)
        terminated = is_success = coverage > self.success_threshold

        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = is_success
        info["coverage"] = coverage

        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup()

        if options is not None and options.get("reset_to_state") is not None:
            state = np.array(options.get("reset_to_state"))
        else:
            # state = self.np_random.uniform(low=[50, 50, 100, 100, -np.pi], high=[450, 450, 400, 400, np.pi])
            rs = np.random.RandomState(seed=seed)
            state = np.array(
                [
                    rs.randint(50, 450),
                    rs.randint(50, 450),
                    rs.randint(100, 400),
                    rs.randint(100, 400),
                    rs.randn() * 2 * np.pi - np.pi,
                ],
                # dtype=np.float64
            )
        self._set_state(state)

        observation = self.get_obs()
        coverage = self._get_coverage()
        info = self._get_info()
        info["is_success"] = False
        info["coverage"] = coverage

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _draw(self):
        # Create a screen
        screen = pygame.Surface((512, 512))
        screen.fill((255, 255, 255))
        draw_options = DrawOptions(screen)

        # Draw goal pose
        goal_body = self.get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [goal_body.local_to_world(v) for v in shape.get_vertices()]
            goal_points = [
                pymunk.pygame_util.to_pygame(point, draw_options.surface)
                for point in goal_points
            ]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(screen, pygame.Color("LightGreen"), goal_points)

        # Draw agent and block
        self.space.debug_draw(draw_options)
        return screen

    def _get_img(self, screen, width, height, render_action=False):
        img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
        img = cv2.resize(img, (width, height))
        render_size = min(width, height)
        if render_action and self._last_action is not None:
            action = np.array(self._last_action)
            coord = (action / 512 * [height, width]).astype(np.int32)
            marker_size = int(8 / 96 * render_size)
            thickness = int(1 / 96 * render_size)
            cv2.drawMarker(
                img,
                coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
            )
        return img

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        screen = self._draw()  # draw the environment on a screen

        if self.render_mode == "rgb_array":
            return self._get_img(
                screen,
                width=width,
                height=height,
                render_action=False,  # NOTE: render the action was set to visualize
            )
        elif self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((512, 512))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.blit(
                screen, screen.get_rect()
            )  # copy our drawings from `screen` to the visible window
            pygame.event.pump()
            self.clock.tick(
                self.metadata["render_fps"] * int(1 / (self.dt * self.control_hz))
            )
            pygame.display.update()
        else:
            raise ValueError(self.render_mode)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def teleop_agent(self):
        teleop_agent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(
                Vec2d(*pygame.mouse.get_pos()), self.screen
            )
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return teleop_agent(act)

    def get_obs(self):
        if self.obs_type == "state":
            agent_position = np.array(self.agent.position)
            block_position = np.array(self.block.position)
            block_angle = self.block.angle % (2 * np.pi)
            return np.concatenate(
                [agent_position, block_position, [block_angle]], dtype=np.float64
            )

        if self.obs_type == "environment_state_agent_pos":
            return {
                "environment_state": self.get_keypoints(self._block_shapes).flatten(),
                "agent_pos": np.array(self.agent.position),
            }

        pixels = self._render()
        if self.obs_type == "pixels":
            return pixels
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": pixels,
                "agent_pos": np.array(self.agent.position),
            }

    @staticmethod
    def get_goal_pose_body(pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = int(1 / self.dt * self.control_hz)
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = self.damping if self.damping is not None else 0.0
        self.teleop = False

        # Add walls
        walls = [
            self.add_segment(self.space, (5, 506), (5, 5), 2),
            self.add_segment(self.space, (5, 5), (506, 5), 2),
            self.add_segment(self.space, (506, 5), (506, 506), 2),
            self.add_segment(self.space, (5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone
        self.agent = self.add_circle(self.space, (256, 400), 15)
        self.block, self._block_shapes = self.add_tee(self.space, (256, 300), 0)
        self.goal_pose = np.array([256, 256, np.pi / 4])  # x, y, theta (in radians)
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog

        # Add collision handling
        self.space.on_collision(0, 0, post_solve=self._handle_collision)
        self.n_contact_points = 0

    def _set_state(self, state):
        self.agent.position = list(state[:2])
        # Setting angle rotates with respect to center of mass, therefore will modify the geometric position if not
        # the same as CoM. Therefore should theoretically set the angle first. But for compatibility with legacy data,
        # we do the opposite.
        self.block.position = list(state[2:4])
        self.block.angle = state[4]

        # Run physics to take effect
        self.space.step(self.dt)

    @staticmethod
    def add_segment(space, a, b, radius):
        # TODO(rcadene): rename add_segment to make_segment, since it is not added to the space
        shape = pymunk.Segment(space.static_body, a, b, radius)
        shape.color = pygame.Color(
            "LightGray"
        )  # https://htmlcolorcodes.com/color-names
        return shape

    @staticmethod
    def add_circle(space, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        space.add(body, shape)
        return body

    @staticmethod
    def add_tee(space, position, angle, scale=30, color="LightSlateGray", mask=None):
        if mask is None:
            mask = pymunk.ShapeFilter.ALL_MASKS()
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.angle = angle
        body.position = position
        body.friction = 1
        space.add(body, shape1, shape2)
        return body, [shape1, shape2]

    @staticmethod
    def get_keypoints(block_shapes):
        """Get a (8, 2) numpy array with the T keypoints.

        The T is composed of two rectangles each with 4 keypoints.

        0───────────1
        │           │
        3───4───5───2
            │   │
            │   │
            │   │
            │   │
            7───6
        """
        keypoints = []
        for shape in block_shapes:
            for v in shape.get_vertices():
                v = v.rotated(shape.body.angle)
                v = v + shape.body.position
                keypoints.append(np.array(v))
        return np.row_stack(keypoints)

    def _get_optimal_action(self, obs):
        raise NotImplementedError(
            "This environment does not support optimal actions. Yet... "
        )


class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        """Draw a pymunk.Space on a pygame.Surface object.

        Typical usage::

        >>> import pymunk
        >>> surface = pygame.Surface((10,10))
        >>> space = pymunk.Space()
        >>> options = pymunk.pygame_util.DrawOptions(surface)
        >>> space.debug_draw(options)

        You can control the color of a shape by setting shape.color to the color
        you want it drawn in::

        >>> c = pymunk.Circle(None, 10)
        >>> c.color = pygame.Color("pink")

        See pygame_util.demo.py for a full example

        Since pygame uses a coordinate system where y points down (in contrast
        to many other cases), you either have to make the physics simulation
        with Pymunk also behave in that way, or flip everything when you draw.

        The easiest is probably to just make the simulation behave the same
        way as Pygame does. In that way all coordinates used are in the same
        orientation and easy to reason about::

        >>> space = pymunk.Space()
        >>> space.gravity = (0, -1000)
        >>> body = pymunk.Body()
        >>> body.position = (0, 0) # will be positioned in the top left corner
        >>> space.debug_draw(options)

        To flip the drawing its possible to set the module property
        :py:data:`positive_y_is_up` to True. Then the pygame drawing will flip
        the simulation upside down before drawing::

        >>> positive_y_is_up = True
        >>> body = pymunk.Body()
        >>> body.position = (0, 0)
        >>> # Body will be position in bottom left corner

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        self.surface = surface
        super().__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(
            self.surface, light_color(fill_color).as_int(), p, round(radius - 4), 0
        )

        # circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        # p2 = to_pygame(circle_edge, self.surface)
        # line_r = 2 if radius > 20 else 1
        # pygame.draw.lines(self.surface, outline_color.as_int(), False, [p, p2], line_r)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]

        radius = 2
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)

        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def get_mouse_pos(surface: pygame.Surface) -> Tuple[int, int]:
    """Get position of the mouse pointer in pymunk coordinates."""
    p = pygame.mouse.get_pos()
    return from_pygame(p, surface)


def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates.

    Note that in case positive_y_is_up is False, this function won't actually do
    anything except converting the point to integers.
    """
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])


def from_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pygame surface local coordinates to
    pymunk coordinates
    """
    return to_pygame(p, surface)


def light_color(color: SpaceDebugColor):
    color = np.minimum(
        1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255])
    )
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color


if __name__ == "__main__":
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

    import xenoworlds

    env = gym.make(
        "xenoworlds/PushT",
        render_mode="rgb_array",
        resolution=224,
    )

    env = gym.wrappers.AddRenderObservation(env, render_only=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)

    xenoworlds.collect.random_action(env, num_episodes=1)
    env = RecordVideo(
        env,
        video_folder="pusht_videos",  # Folder to save videos
        name_prefix="eval",  # Prefix for video filenames
        episode_trigger=lambda x: True,  # Record every episode
    )

    observation, info = env.reset()  #
    print(observation)
    print(info)

    fig, axs = plt.subplots(1, 1)
    rgb_array = env.unwrapped.render()  #
    axs.imshow(rgb_array)
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_title("pusht.")
    plt.savefig("pusht.png")
    plt.close()  # Close the plot to free up memory
    for i in range(5):
        action = env.unwrapped.action_space.sample()  # Sample a random action
        # _get_optimal_action()
        env.step(action)

    print("Saved PushT observation as pusht.png")

    # 5. Close the environment
    env.close()
