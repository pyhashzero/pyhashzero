from gym.envs.registration import register

from .cube import *
from .puzzle import *
from .sokoban import *
from .tiles import *

register(
    'Sokoban-v0',
    entry_point='hzai.env.sokoban:SokobanEnv',
    kwargs={
        'xml': 'default.xml',
        'xmls': 'env/assets/sokoban/xmls',
        'sprites': 'env/assets/sokoban/sprites'
    }
)
