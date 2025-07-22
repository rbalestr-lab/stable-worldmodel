import os

from . import planner
from .world import World
from .agent import Agent
from .wm import DummyWorldModel
from . import wrappers
from .utils import set_state, create_pil_image_from_url
from . import data
from . import collect
from gymnasium.envs.registration import register

register(
    id="xenoworlds/ImagePositioning-v1",
    entry_point="xenoworlds.envs.image_positioning:ImagePositioning",
)

register(
    id="xenoworlds/PushT-v1",
    entry_point="xenoworlds.envs.pusht:PushT",
)
