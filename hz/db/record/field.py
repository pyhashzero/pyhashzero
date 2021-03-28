from hz.serializable import Serializable

__all__ = ['Field']


class Field(Serializable):
    def __init__(self, name, kind, nullable=True, fallback=None):
        super(Field, self).__init__()

        self.name = name
        self.kind = kind
        self.nullable = nullable
        self.fallback = fallback
