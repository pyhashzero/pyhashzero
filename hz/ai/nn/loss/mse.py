import numpy as np

from hz.ai.nn.core import Loss


class MSELoss(Loss):
    def __init__(self):
        super(MSELoss, self).__init__()

        self._loss = None

    def forward(self, prediction, target) -> np.array:
        self._loss = (((prediction - target) ** 2) / np.prod(target.shape)).sum()
        return self._loss
