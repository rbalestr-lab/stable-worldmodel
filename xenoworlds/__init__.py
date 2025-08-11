import os

from gymnasium.envs.registration import register

from . import collect, data, policy, solver, wrappers, predictor
from .utils import create_pil_image_from_url, set_state
from .wm import DummyWorldModel
from .world import World
from .evaluator import Evaluator

register(
    id="xenoworlds/ImagePositioning-v1",
    entry_point="xenoworlds.envs.image_positioning:ImagePositioning",
)

register(
    id="xenoworlds/PushT-v1",
    entry_point="xenoworlds.envs.pusht:PushT",
)
