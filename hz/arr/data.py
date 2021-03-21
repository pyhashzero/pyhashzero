__all__ = ['boolean', 'integer', 'int16', 'int32', 'int64', 'floating', 'float16', 'float32', 'float64']


class Generic(object):
    """
    Base class for numpy scalar types.

        Class from which most (all?) numpy scalar types are derived.  For
        consistency, exposes the same API as `ndarray`, despite many
        consequent attributes being either "get-only," or completely irrelevant.
        This is the class from which it is strongly suggested users should derive
        custom scalar types.
    """

    def __init__(self, data):
        self._data = data

    def __abs__(self, *args, **kwargs):
        raise NotImplementedError

    def __add__(self, *args, **kwargs):
        raise NotImplementedError

    def __and__(self, *args, **kwargs):
        raise NotImplementedError

    def __bool__(self, *args, **kwargs):
        raise NotImplementedError

    def __copy__(self, *args, **kwargs):
        raise NotImplementedError

    def __deepcopy__(self, *args, **kwargs):
        raise NotImplementedError

    def __divmod__(self, *args, **kwargs):
        raise NotImplementedError

    def __eq__(self, *args, **kwargs):
        raise NotImplementedError

    def __float__(self, *args, **kwargs):
        raise NotImplementedError

    def __floordiv__(self, *args, **kwargs):
        raise NotImplementedError

    def __format__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, *args, **kwargs):
        raise NotImplementedError

    def __ge__(self, *args, **kwargs):
        raise NotImplementedError

    def __gt__(self, *args, **kwargs):
        raise NotImplementedError

    def __hash__(self, *args, **kwargs):
        raise NotImplementedError

    def __int__(self):
        raise NotImplementedError

    def __invert__(self, *args, **kwargs):
        raise NotImplementedError

    def __le__(self, *args, **kwargs):
        raise NotImplementedError

    def __lshift__(self, *args, **kwargs):
        raise NotImplementedError

    def __lt__(self, *args, **kwargs):
        raise NotImplementedError

    def __mod__(self, *args, **kwargs):
        raise NotImplementedError

    def __mul__(self, *args, **kwargs):
        raise NotImplementedError

    def __neg__(self, *args, **kwargs):
        raise NotImplementedError

    def __ne__(self, *args, **kwargs):
        raise NotImplementedError

    def __or__(self, *args, **kwargs):
        raise NotImplementedError

    def __pos__(self, *args, **kwargs):
        raise NotImplementedError

    def __pow__(self, *args, **kwargs):
        raise NotImplementedError

    def __radd__(self, *args, **kwargs):
        raise NotImplementedError

    def __rand__(self, *args, **kwargs):
        raise NotImplementedError

    def __rdivmod__(self, *args, **kwargs):
        raise NotImplementedError

    def __reduce__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self, *args, **kwargs):
        raise NotImplementedError

    def __rfloordiv__(self, *args, **kwargs):
        raise NotImplementedError

    def __rlshift__(self, *args, **kwargs):
        raise NotImplementedError

    def __rmod__(self, *args, **kwargs):
        raise NotImplementedError

    def __rmul__(self, *args, **kwargs):
        raise NotImplementedError

    def __ror__(self, *args, **kwargs):
        raise NotImplementedError

    def __round__(self, *args, **kwargs):
        raise NotImplementedError

    def __rpow__(self, *args, **kwargs):
        raise NotImplementedError

    def __rrshift__(self, *args, **kwargs):
        raise NotImplementedError

    def __rshift__(self, *args, **kwargs):
        raise NotImplementedError

    def __rsub__(self, *args, **kwargs):
        raise NotImplementedError

    def __rtruediv__(self, *args, **kwargs):
        raise NotImplementedError

    def __rxor__(self, *args, **kwargs):
        raise NotImplementedError

    def __setstate__(self, *args, **kwargs):
        raise NotImplementedError

    def __sizeof__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self, *args, **kwargs):
        raise NotImplementedError

    def __sub__(self, *args, **kwargs):
        raise NotImplementedError

    def __truediv__(self, *args, **kwargs):
        raise NotImplementedError

    def __xor__(self, *args, **kwargs):
        raise NotImplementedError

    def all(self, *args, **kwargs):
        raise NotImplementedError

    def any(self, *args, **kwargs):
        raise NotImplementedError

    def argmax(self, *args, **kwargs):
        raise NotImplementedError

    def argmin(self, *args, **kwargs):
        raise NotImplementedError

    def argsort(self, *args, **kwargs):
        raise NotImplementedError

    def astype(self, *args, **kwargs):
        raise NotImplementedError

    def choose(self, *args, **kwargs):
        raise NotImplementedError

    def clip(self, *args, **kwargs):
        raise NotImplementedError

    def copy(self, *args, **kwargs):
        raise NotImplementedError

    def diagonal(self, *args, **kwargs):
        raise NotImplementedError

    def dump(self, *args, **kwargs):
        raise NotImplementedError

    def dumps(self, *args, **kwargs):
        raise NotImplementedError

    def fill(self, *args, **kwargs):
        raise NotImplementedError

    def flatten(self, *args, **kwargs):
        raise NotImplementedError

    def max(self, *args, **kwargs):
        raise NotImplementedError

    def mean(self, *args, **kwargs):
        raise NotImplementedError

    def min(self, *args, **kwargs):
        raise NotImplementedError

    def nonzero(self, *args, **kwargs):
        raise NotImplementedError

    def prod(self, *args, **kwargs):
        raise NotImplementedError

    def repeat(self, *args, **kwargs):
        raise NotImplementedError

    def reshape(self, *args, **kwargs):
        raise NotImplementedError

    def resize(self, *args, **kwargs):
        raise NotImplementedError

    def round(self, *args, **kwargs):
        raise NotImplementedError

    def sort(self, *args, **kwargs):
        raise NotImplementedError

    def squeeze(self, *args, **kwargs):
        raise NotImplementedError

    def std(self, *args, **kwargs):
        raise NotImplementedError

    def sum(self, *args, **kwargs):
        raise NotImplementedError

    def swapaxes(self, *args, **kwargs):
        raise NotImplementedError

    def take(self, *args, **kwargs):
        raise NotImplementedError

    def tobytes(self, *args, **kwargs):
        raise NotImplementedError

    def tofile(self, *args, **kwargs):
        raise NotImplementedError

    def tolist(self, *args, **kwargs):
        raise NotImplementedError

    def tostring(self, *args, **kwargs):
        raise NotImplementedError

    def transpose(self, *args, **kwargs):
        raise NotImplementedError

    def var(self, *args, **kwargs):
        raise NotImplementedError

    def view(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def data(self):
        return self._data


class Bool(Generic):
    def __init__(self, data):
        if isinstance(data, bool):
            data = data
        elif isinstance(data, Bool):
            data = data.data
        elif isinstance(data, (int, bool)):
            data = bool(data)
        elif isinstance(data, Number):
            data = bool(data.data)
        else:
            raise ValueError(f'{type(data)} cannot be converted to Bool')

        super(Bool, self).__init__(data)

    def __bool__(self):
        return bool(self.data)

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.data)

    def __invert__(self):
        return Bool(~self.data)

    def __copy__(self):
        return Bool(self.data)

    def __deepcopy__(self, memodict=None):
        return Bool(self.data)

    def __and__(self, other):
        if isinstance(other, bool):
            return Bool(self.data & other)
        elif isinstance(other, Bool):
            return Bool(self.data & other.data)
        elif isinstance(other, Number):
            return Bool(self.data & bool(other.data))
        elif isinstance(other, (int, bool)):
            return Bool(self.data & bool(other))
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __or__(self, other):
        if isinstance(other, bool):
            return Bool(self.data | other)
        elif isinstance(other, Bool):
            return Bool(self.data | other.data)
        elif isinstance(other, Number):
            return Bool(self.data | bool(other.data))
        elif isinstance(other, (int, bool)):
            return Bool(self.data | bool(other))
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __xor__(self, other):
        if isinstance(other, bool):
            return Bool(self.data ^ other)
        elif isinstance(other, Bool):
            return Bool(self.data ^ other.data)
        elif isinstance(other, Number):
            return Bool(self.data ^ bool(other.data))
        elif isinstance(other, (int, bool)):
            return Bool(self.data ^ bool(other))
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __eq__(self, other):
        if isinstance(other, bool):
            return Bool(self.data == other)
        elif isinstance(other, Bool):
            return Bool(self.data == other.data)
        elif isinstance(other, Number):
            return Bool(self.data == bool(other.data))
        elif isinstance(other, (int, bool)):
            return Bool(self.data == bool(other))
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __ne__(self, other):
        if isinstance(other, bool):
            return Bool(self.data != other)
        elif isinstance(other, Bool):
            return Bool(self.data != other.data)
        elif isinstance(other, Number):
            return Bool(self.data != bool(other.data))
        elif isinstance(other, (int, bool)):
            return Bool(self.data != bool(other))
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def copy(self):
        return Bool(self.data)


class Number(Generic):
    def __init__(self, data):
        if isinstance(data, Number):
            data = data.data
        elif isinstance(data, (int, float)):
            data = data
        else:
            raise ValueError(f'{type(data)} cannot be converted to Int')

        super(Number, self).__init__(data)

    def __bool__(self):
        return bool(self.data)

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.data)

    def __abs__(self):
        return type(self)(self.data)

    def __invert__(self):
        return type(self)(~self.data)

    def __neg__(self):
        return type(self)(-self.data)

    def __copy__(self):
        return type(self)(self.data)

    def __deepcopy__(self, memodict=None):
        return type(self)(self.data)

    def __and__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data & other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data & other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __or__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data | other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data | other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __xor__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data ^ other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data ^ other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __eq__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data == other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data == other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __ne__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data != other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data != other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __ge__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data >= other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data >= other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __gt__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data > other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data > other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __le__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data <= other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data <= other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __lt__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data < other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data < other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __add__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data + other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data + other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __sub__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data - other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data - other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __mul__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data * other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data * other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __floordiv__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data // other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data // other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __truediv__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data / other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data / other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __divmod__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data // other.data), type(self)(self.data % other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data // other), type(self)(self.data % other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __mod__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data % other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data % other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __pow__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data ** other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data ** other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __lshift__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data << other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data << other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def __rshift__(self, other):
        if isinstance(other, Number):
            return type(self)(self.data >> other.data)
        elif isinstance(other, (int, float)):
            return type(self)(self.data >> other)
        else:
            raise ValueError(f'{type(other)} is not compatible with {type(self)}')

    def copy(self):
        return type(self)(self.data)


class Int(Number):
    def __init__(self, data):
        if isinstance(data, bool):
            data = int(data)
        elif isinstance(data, Bool):
            data = int(data.data)
        elif isinstance(data, (int, float)):
            data = int(data)
        elif isinstance(data, Number):
            data = int(data.data)
        else:
            raise ValueError(f'{type(data)} cannot be converted to Int')

        super(Int, self).__init__(data)


class Float(Number):
    def __init__(self, data):
        if isinstance(data, bool):
            data = float(data)
        elif isinstance(data, Bool):
            data = float(data.data)
        elif isinstance(data, (int, float)):
            data = float(data)
        elif isinstance(data, Number):
            data = float(data.data)
        else:
            raise ValueError(f'{type(data)} cannot be converted to Int')

        super(Float, self).__init__(data)


boolean = Bool

integer = Int
int16 = Int
int32 = Int
int64 = Int

floating = Float
float16 = Float
float32 = Float
float64 = Float
