from ... import functional as f
from ...core import Module


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inp):
        return f.sigmoid(
            inp=inp
        )
