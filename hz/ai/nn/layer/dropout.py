from .. import functional as f


class Dropout:
    def __init__(self, keep_prob):
        self._keep_prob = keep_prob

    def forward(self, inp):
        return f.dropout(
            inp=inp,
            keep_prob=self._keep_prob
        )
