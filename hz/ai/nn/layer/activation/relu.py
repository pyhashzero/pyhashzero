from ... import functional as f
from ...core import Module


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, inp):
        return f.relu(
            inp=inp
        )
