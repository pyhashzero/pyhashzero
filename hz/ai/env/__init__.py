from gym.envs.registration import register

from .cube import CubeEnv
from .puzzle import Puzzle2048
from .sokoban import SokobanEnv
from .tiles import TilesEnv

register(
    'Sokoban-v0',
    entry_point='hzai.env.sokoban:SokobanEnv',
    kwargs={
        'xml': 'default.xml',
        'xmls': 'assets/sokoban/xmls',
        'sprites': 'assets/sokoban/sprites'
    }
)
