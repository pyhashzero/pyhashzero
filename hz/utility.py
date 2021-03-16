import json
import re
import typing
import uuid
from datetime import (
    date,
    datetime,
    time,
    timedelta
)
from time import mktime, struct_time

pascal_case_pattern = re.compile(r'(?<!^)(?=[A-Z])')
# camel_case_pattern = re.compile(r'(\b[a-z]+|\G(?!^))((?:[A-Z]|\d+)[a-z]*)')
email_pattern = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
ip_address_pattern = re.compile(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
                                r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
                                r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
                                r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")


def empty_object_id():
    return uuid.UUID('00000000-0000-0000-0000-000000000000')


def new_object_id():
    return uuid.uuid4()


def current_time():
    return datetime.now()


def new_nickname():
    pass


def new_password():
    pass


def from_camel_to_pascal(string: str):
    return string


def from_camel_to_snake(string: str):
    return string


def from_pascal_to_camel(string: str):
    return string


def from_pascal_to_snake(string: str):
    return string


def from_snake_to_camel(string: str):
    return string


def from_snake_to_pascal(string: str):
    return string


def to_snake_string(string: str):
    return pascal_case_pattern.sub('_', string).lower()


def to_pascal_string(string: str):
    return ''.join(word.title() for word in string.split('_'))


def to_upper_string(string: str):
    return string.upper()


def to_lower_string(string: str):
    return string.lower()


def to_title(string: str):
    return string.title()


def to_enumerable(value):
    if isinstance(value, dict):
        ret = {k: to_enumerable(v) for k, v in value.items()}
    elif isinstance(value, list):
        ret = [to_enumerable(v) for v in value]
    elif isinstance(value, tuple):
        ret = tuple([to_enumerable(v) for v in value])
    elif type(value).__module__.startswith('core') or type(value).__module__.startswith('model'):
        ret = to_enumerable({k: v for k, v in value.__dict__.items() if not k.startswith('_')})
    else:
        ret = value
    return ret


def to_case(case, value, key=None, convert_value=False):
    enumerable = to_enumerable(value)

    if case == 'snake':
        new_key = to_snake_string(key) if key is not None else None
    elif case == 'pascal':
        new_key = to_pascal_string(key) if key is not None else None
    elif case == 'upper':
        new_key = to_upper_string(key) if key is not None else None
    elif case == 'lower':
        new_key = to_lower_string(key) if key is not None else None
    else:
        raise ValueError(f'{case} is not supported case.')

    if isinstance(enumerable, dict):
        ret = new_key, {k: v for k, v in [to_case(case, v, k) for k, v in enumerable.items()]}
    elif isinstance(enumerable, list):
        ret = new_key, [to_case(case, v) for v in enumerable]
    elif isinstance(enumerable, tuple):
        ret = new_key, tuple([to_case(case, v) for v in enumerable])
    elif type(value).__module__.startswith('auth') or type(value).__module__.startswith('user'):
        ret = new_key, to_case(case, value.__dict__)
    else:
        if convert_value:
            value = to_snake_string(value)
        ret = new_key, value

    if new_key is None:
        return ret[1]
    return ret


def to_hash(func, *args, **kwargs):
    return f'{func.__module__}.{func.__name__}({", ".join(f"{arg}" for arg in args)}, {", ".join([f"{k}={v}" for k, v in kwargs.items()])})'


def get_converter(name):
    def _dummy(_: object) -> object:
        raise ValueError(f'could not found the appropriate converter')

    def _datetime_converter(value: object) -> typing.Optional[datetime]:
        if value is None:
            return value

        if isinstance(value, datetime):
            return value

        if isinstance(value, struct_time):
            return datetime.fromtimestamp(mktime(value))

        if isinstance(value, str):
            return datetime.fromisoformat(value)

        raise ValueError(f'cannot convert {value} with type {type(value)} to datetime')

    def _integer_converter(value: object) -> typing.Optional[int]:
        if value is None:
            return value

        if isinstance(value, int):
            return value

        if isinstance(value, str):
            return int(value)

        raise ValueError(f'cannot convert {value} with type {type(value)} to integer')

    def _float_converter(value: object) -> typing.Optional[float]:
        if value is None:
            return value

        if isinstance(value, float):
            return value

        if isinstance(value, str):
            return float(value)

        raise ValueError(f'cannot convert {value} with type {type(value)} to float')

    def _string_converter(value: object) -> typing.Optional[str]:
        if value is None:
            return value

        if isinstance(value, str):
            return value

        raise ValueError(f'cannot convert {value} with type {type(value)} to string')

    def _byte_converter(value: object) -> typing.Optional[bytes]:
        if value is None:
            return value

        if isinstance(value, bytes):
            return value

        raise ValueError(f'cannot convert {value} with type {type(value)} to bytes')

    def _boolean_converter(value: object) -> typing.Optional[bool]:
        if value is None:
            return value

        if isinstance(value, bool):
            return value

        raise ValueError(f'cannot convert {value} with type {type(value)} to boolean')

    def _uuid_converter(value: object) -> typing.Optional[uuid.UUID]:
        if value is None:
            return value

        if isinstance(value, uuid.UUID):
            return value

        if isinstance(value, str):
            return uuid.UUID(value)

        raise ValueError(f'cannot convert {value} with type {type(value)} to uuid')

    def _dictionary_converter(value: object) -> typing.Optional[dict]:
        if value is None:
            return value

        if isinstance(value, dict):
            return to_enumerable(value)

        if hasattr(value, 'dict'):
            return to_enumerable(value)

        raise ValueError(f'cannot convert {value} with type {type(value)} to dictionary')

    def _list_converter(value: object) -> typing.Optional[list]:
        if value is None:
            return value

        if isinstance(value, list):
            ret = []
            for v in value:
                ret.append(to_enumerable(v))
            return ret

        raise ValueError(f'cannot convert {value} with type {type(value)} to list')

    def _tuple_converter(value: object) -> typing.Optional[tuple]:
        if value is None:
            return value

        if isinstance(value, tuple):
            ret = []
            for v in value:
                ret.append(to_enumerable(v))
            ret = tuple(ret)
            return ret

        raise ValueError(f'cannot convert {value} with type {type(value)} to tuple')

    def _object_converter(value: object) -> typing.Optional[object]:
        if value is None:
            return value

        if isinstance(value, dict):
            return _dictionary_converter(value)

        if isinstance(value, list):
            return _list_converter(value)

        if isinstance(value, tuple):
            return _tuple_converter(value)

        if hasattr(value, 'dict'):
            return value.dict

        return value

    converters = {
        'datetime': _datetime_converter,
        'int': _integer_converter,
        'integer': _integer_converter,
        'float': _float_converter,
        'str': _string_converter,
        'string': _string_converter,
        'byte': _byte_converter,
        'bytes': _byte_converter,
        'bool': _boolean_converter,
        'boolean': _boolean_converter,
        'uuid': _uuid_converter,
        'dict': _dictionary_converter,
        'dictionary': _dictionary_converter,
        'list': _list_converter,
        'tuple': _tuple_converter,
        'object': _object_converter,
        'enumerable': _object_converter,
    }

    return converters.get(name.lower(), _dummy)


def get_type(name):
    type_mapper = {
        'datetime': 'datetime',
        'int': 'int',
        'integer': 'int',
        'float': 'float',
        'str': 'str',
        'string': 'str',
        'byte': 'byte',
        'bytes': 'byte',
        'bool': 'bool',
        'boolean': 'bool',
        'uuid': 'uuid',
        'dict': 'dict',
        'dictionary': 'dict',
        'list': 'list',
        'tuple': 'tuple',
        'object': 'object',
        'enumerable': 'enumerable',
    }

    return type_mapper.get(name.lower(), 'unknown')


def normalize_kwargs(document_type, **kwargs):
    if hasattr(document_type, 'fields'):
        ret = {}

        fields = document_type.fields()
        for key in kwargs.keys():
            field = list(filter(lambda x: x.name == key, fields.values()))
            if len(field) != 1:
                raise ValueError(f'field {key} has to be only one on the document')
            field = field[0]

            ret[key] = get_converter(field.kind)(kwargs[key])
        return ret
    else:
        return kwargs


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return (datetime.min + obj).time().isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if callable(obj):
            return str(obj)
        return obj
