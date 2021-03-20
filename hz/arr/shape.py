class Shape:
    def __init__(self, *shape):
        if len(shape) == 0 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        self._shape = shape

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'({", ".join([str(x) for x in self._shape])})'

    def __len__(self):
        return len(self._shape)

    def __iter__(self):
        for s in self._shape:
            yield s

    def tuple(self):
        return tuple(self._shape)

    def list(self):
        return list(self._shape)
