from collections import OrderedDict
from datetime import datetime

import pymongo
from pymongo import (
    read_concern,
    write_concern
)

from .core import (
    Database,
    Transaction
)
from .record.constraint import (
    CustomConstraint,
    UniqueConstraint
)
from .record.document import Document
from ..utility import (
    empty_object_id,
    normalize_kwargs
)


class MongoTransaction(Transaction):
    def __init__(self, session):
        super(MongoTransaction, self).__init__()

        self.session = session

    def start(self):
        self.session.start_transaction(read_concern.ReadConcern('snapshot'), write_concern.WriteConcern('majority'))

    def commit(self):
        self.session.commit_transaction()

    def abort(self):
        self.session.abort_transaction()

    def end(self):
        self.session.end_session()


class MongoDatabase(Database):
    def __init__(self, connection: str):
        super(MongoDatabase, self).__init__()

        self._client = pymongo.MongoClient(connection, replicaset='rs0')
        self._database = self._client['H0']

    def drop(self, transaction=None):
        # drop only documents that is in the namespace
        database = self._database if transaction is None else transaction.session.client.H0

        for collection_name in database.list_collection_names():
            database[collection_name].drop()

    def initialize(self, transaction=None):
        def get_type(name):
            type_mapper = {
                'datetime': 'date',
                'int': 'int',
                'integer': 'int',
                'float': 'double',
                'str': 'string',
                'string': 'string',
                'byte': 'binData',
                'bytes': 'binData',
                'bool': 'bool',
                'boolean': 'bool',
                'uuid': 'binData',
                'dict': 'object',
                'dictionary': 'object',
                'list': 'array',
                'tuple': 'array',
                'object': 'object',
                'enumerable': 'object'
            }
            return type_mapper.get(name.lower(), 'unknown')

        database = self._database if transaction is None else transaction.session.client.H0

        for collection_name, document in Document.subclasses().items():
            if collection_name != 'arxiv.source_file':
                continue

            database.create_collection(collection_name)

            vexpr = {
                '$jsonSchema': {
                    'bsonType': 'object',
                    'required': list(map(lambda x: x.name, filter(lambda y: not y.nullable, document.fields().values()))),
                    'properties': {}
                }
            }
            for field in document.fields().values():
                _ = list(filter(lambda x: x.name == field.name and not isinstance(x, (UniqueConstraint, CustomConstraint)), document.constraints()))

                vexpr['$jsonSchema']['properties'][field.name] = {
                    'bsonType': get_type(field.kind) if not field.nullable else [get_type(field.kind), 'null'],
                    'description': ''
                }

            cmd = OrderedDict([
                ('collMod', document.CollectionName),
                ('validator', vexpr),
                ('validationLevel', 'moderate')
            ])

            database.command(cmd)

            used_fields = set()
            for constraint in document.constraints():
                if isinstance(constraint, UniqueConstraint):
                    if ',' in constraint.field_name:
                        constraint_fields = list(map(lambda x: x.strip(), constraint.field_name.split(',')))
                    else:
                        constraint_fields = [constraint.field_name]

                    for constraint_field in constraint_fields:
                        used_fields.add(constraint_field)

                    c = [(f'{k}', 1) for k in constraint_fields]
                    database[collection_name].create_index(c, name=f'unique_constraint_{"_".join(constraint_fields)}', unique=True)

            for index in document.indexes():
                if ',' in index.field_name:
                    index_fields = list(map(lambda x: x.strip(), index.field_name.split(',')))
                else:
                    index_fields = [index.field_name]

                if any(list(map(lambda x: x in used_fields, index_fields))):
                    continue

                c = [(f'{k}', 1) for k in index_fields]
                database[collection_name].create_index(c, name=f'index_{"_".join(index_fields)}')

    def save(self, document, transaction=None):
        if not isinstance(document, Document):
            raise ValueError(f'{type(document)} is not valid for saving')

        database = self._database if transaction is None else transaction.session.client.H0
        session = None if transaction is None else transaction.session

        document.creator_id = empty_object_id()
        document.created_at = datetime.utcnow()
        document.updater_id = empty_object_id()
        document.updated_at = datetime.utcnow()

        document.validate()

        collection = database[document.CollectionName]
        collection.insert_one(document.write_dict, session=session)

        return document

    def save_many(self, *documents, transaction=None):
        ret = []
        for document in documents:
            ret.append(self.save(document, transaction))
        return ret

    def update(self, document, transaction=None):
        if not isinstance(document, Document):
            raise ValueError(f'{type(document)} is not valid for saving')

        database = self._database if transaction is None else transaction.session.client.H0
        session = None if transaction is None else transaction.session

        document.updater_id = empty_object_id()
        document.updated_at = datetime.utcnow()

        document.validate()

        query = {'object_id': document.object_id}
        update = {'$set': document.write_dict}

        collection = database[document.CollectionName]
        collection.update_one(query, update, session=session)

        return document

    def delete(self, document, transaction=None):
        if not isinstance(document, Document):
            raise ValueError(f'{type(document)} is not valid for saving')

        document.updater_id = empty_object_id()
        document.updated_at = datetime.utcnow()
        document.deleter_id = empty_object_id()
        document.deleted_at = datetime.utcnow()
        document.is_deleted = True
        return self.update(document, transaction)

    def read_one(self, document_type, transaction=None, **kwargs):
        database = self._database if transaction is None else transaction.session.client.H0
        session = None if transaction is None else transaction.session

        if issubclass(document_type, Document):
            collection = database[document_type.CollectionName]
        else:
            raise ValueError()

        result = collection.find_one(normalize_kwargs(document_type, **kwargs), {'_id': 0}, session=session)

        if result is None:
            return None
        return document_type.from_dict(result)

    def read_many(self, document_type, transaction=None, **kwargs):
        database = self._database if transaction is None else transaction.session.client.H0
        session = None if transaction is None else transaction.session

        if issubclass(document_type, Document):
            collection = database[document_type.CollectionName]
        else:
            raise ValueError()

        result = collection.find(normalize_kwargs(document_type, **kwargs), {'_id': 0}, session=session)

        for document in result:
            yield document_type.from_dict(document)

    def count(self, document_type, transaction=None, **kwargs):
        database = self._database if transaction is None else transaction.session.client.H0
        session = None if transaction is None else transaction.session

        if issubclass(document_type, Document):
            collection = database[document_type.CollectionName]
        else:
            raise ValueError()

        return collection.count_documents(normalize_kwargs(document_type, **kwargs), session=session)

    def new_session(self):
        session = self._client.start_session()
        return MongoTransaction(session)
