from hz.core import CoreObject


class Cache(CoreObject):
    def __init__(self):
        super(Cache, self).__init__()

    def remove_all(self):
        raise NotImplementedError

    def add(self, key, value):
        raise NotImplementedError

    def update(self, key, value):
        raise NotImplementedError

    def remove(self, key):
        raise NotImplementedError

    def get(self, key):
        raise NotImplementedError

    def has(self, key):
        raise NotImplementedError
