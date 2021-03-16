import redis

from hz.core import CoreObject


class RedisCache(CoreObject):
    def __init__(self, connection: str):
        super(RedisCache, self).__init__()

        self._cache = redis.Redis(connection_pool=redis.BlockingConnectionPool.from_url(connection))

    def remove_all(self):
        self._cache.flushall()

    def add(self, key, value):
        self._cache.set(key, value)

    def update(self, key, value):
        self._cache.set(key, value)

    def remove(self, key):
        self._cache.delete(key)

    def get(self, key):
        return self._cache.get(key)

    def has(self, key):
        return self._cache.exists(key)
