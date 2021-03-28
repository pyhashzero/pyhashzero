from hz.serializable import Serializable
from hz.utility import (
    current_time,
    get_converter,
    get_type,
    new_object_id
)
from .constraint import UniqueConstraint
from .field import Field
from .index import Index

__all__ = ['Document']


class Document(Serializable):
    CollectionName = None

    Fields = [
        Field('object_id', 'UUID', nullable=False, fallback=new_object_id),
        Field('created_at', 'Datetime', nullable=False, fallback=current_time),
        Field('creator_id', 'UUID', nullable=False),
        Field('updated_at', 'Datetime', nullable=False, fallback=current_time),
        Field('updater_id', 'UUID', nullable=False),
        Field('deleted_at', 'Datetime'),
        Field('deleter_id', 'UUID'),
        Field('is_deleted', 'Boolean', nullable=False, fallback=False)
    ]

    Constraints = [
        UniqueConstraint('object_id')
    ]

    Indexes = [
        Index('object_id')
    ]

    @classmethod
    def fields(cls) -> dict:
        fields = {}

        for x in cls.mro()[::-1]:
            if hasattr(x, 'Fields'):
                _fields = getattr(x, 'Fields')
                for _field in _fields:
                    fields[_field.name] = _field

        return fields

    @classmethod
    def constraints(cls) -> list:
        constraints = []

        for x in cls.mro()[::-1]:
            if hasattr(x, 'Constraints'):
                _constraints = getattr(x, 'Constraints')
                constraints.extend(_constraints)

        return constraints

    @classmethod
    def indexes(cls) -> list:
        indexes = []

        for x in cls.mro()[::-1]:
            if hasattr(x, 'Indexes'):
                _indexes = getattr(x, 'Indexes')
                indexes.extend(_indexes)

        return indexes

    @classmethod
    def subclasses(cls) -> dict:
        subclasses = {}
        for x in cls.__subclasses__():
            if x.CollectionName in subclasses:
                raise ValueError(f'{x.CollectionName} is already in subclasses')
            subclasses[x.CollectionName] = x

        return subclasses

    @property
    def read_dict(self):
        return super(Document, self).dict

    @property
    def write_dict(self):
        if self._dirty:
            raise ValueError('you have to validate the document before creating a dictionary from document')

        d = {}
        document_dict = super(Document, self).dict
        for key, value in document_dict.items():
            if key in type(self).fields().keys():
                d[key] = value

        return d

    def __init__(self, **kwargs):
        super(Document, self).__init__(**kwargs)

        self._dirty = True

        for field_name, field in type(self).fields().items():
            if field_name not in kwargs:
                setattr(self, field_name, None)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            self.__dict__[key] = value
            return

        if key not in self.fields().keys():
            self.__dict__[key] = value
            return

        fields = type(self).fields()
        if key not in fields.keys():
            raise ValueError(f'field {key} is not in document fields')

        field = fields[key]
        self.__dict__[key] = get_converter(field.kind)(value)

        self._dirty = True

    def validate(self):
        for field_name, field in type(self).fields().items():
            field_value = getattr(self, field_name, None)

            default_value = None
            if field.fallback is not None and callable(field.fallback):
                default_value = field.fallback()
            elif field.fallback is not None and not callable(field.fallback):
                default_value = field.fallback

            # field value = field value or default value
            # check constraints
            # set field value

            if field_value is None and not field.nullable:
                setattr(self, field_name, default_value)

            field_value = getattr(self, field_name, None)

            if field_value is None and not field.nullable:
                raise ValueError(f'field {field_name} is not nullable')

            if ((field_value is not None and field.nullable) or not field.nullable) and get_type(type(field_value).__name__) != get_type(field.kind):
                raise ValueError(f'field {field_name} has to be type {field.kind} not {type(field_value).__name__}')

            field_constraints = list(filter(lambda x: x.field_name == field_name, type(self).constraints()))

            if not field.nullable and field_constraints is not None:
                for constraint in field_constraints:
                    constraint.check(field_value)

        self._dirty = False
