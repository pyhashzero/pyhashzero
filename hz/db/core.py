from hz.core import CoreObject

__all__ = ['Transaction', 'Database']


class Transaction(CoreObject):
    def __init__(self):
        pass

    def commit(self):
        raise NotImplementedError

    def abort(self):
        raise NotImplementedError

    def end(self):
        raise NotImplementedError


class Database(CoreObject):
    def __init__(self):
        super(Database, self).__init__()

    def drop(self, transaction=None):
        raise NotImplementedError

    def initialize(self, transaction=None):
        raise NotImplementedError

    def save(self, document, transaction=None):
        raise NotImplementedError

    def save_many(self, *documents, transaction=None):
        raise NotImplementedError

    def update(self, document, transaction=None):
        raise NotImplementedError

    def delete(self, document, transaction=None):
        raise NotImplementedError

    def read_one(self, document_type, transaction=None, **kwargs):
        raise NotImplementedError

    def read_many(self, document_type, transaction=None, **kwargs):
        raise NotImplementedError

    def count(self, document_type, transaction=None, **kwargs):
        raise NotImplementedError

    def new_session(self):
        raise NotImplementedError
