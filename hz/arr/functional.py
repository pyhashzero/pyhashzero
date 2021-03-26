import math
from typing import (
    List,
    Tuple,
    Union
)

from hz.arr.array import Array
from hz.arr.data import *
from hz.arr.shape import Shape

__all__ = [
    'absolute', 'add', 'amax', 'amin', 'arange', 'argmax', 'argmin', 'around', 'asarray', 'astype', 'ceil',
    'clip', 'concatenate', 'copy', 'dim', 'dot', 'empty', 'eq', 'exp', 'expand_dims', 'eye', 'fill', 'flatten',
    'floor', 'floordiv', 'full', 'ge', 'getitem', 'gt', 'indices', 'le', 'linspace', 'lt', 'mean',
    'median', 'mul', 'ne', 'negative', 'ones', 'ones_like', 'pad', 'power', 'prod', 'put_along_axis', 'repeat',
    'reshape', 'setitem', 'size', 'split', 'sqrt', 'square', 'squeeze', 'stack', 'std', 'sub', 'sum',
    'take_along_axis', 'tanh', 'tolist', 'transpose', 'truediv', 'unique', 'var', 'where', 'zeros', 'zeros_like'
]

BooleanT = Union[bool, boolean]
IntegerT = Union[int, integer, int16, int32, int64]
FloatingT = Union[float, floating, float16, float32, float64]
NumberT = Union[IntegerT, FloatingT]

DataT = Union[BooleanT, IntegerT, FloatingT]
ArrayT = Union[Tuple[DataT], List[DataT], Array, Tuple['ArrayT'], List['ArrayT']]
TypeT = Union[str]
ShapeT = Union[Tuple[int], List[int], Shape]
IndexT = Union[int, slice, Tuple[Union[int, slice], ...]]


class Broadcast:
    def __init__(self, inp1: ArrayT, inp2: ArrayT):
        if not isinstance(inp2, Array):
            inp2 = Array(inp2)

        # both parameters are float, int, bool
        if inp1.ndim == 0 and inp2.ndim == 0:
            pass

        # first parameter is NDArray, second parameter is float, int, bool
        if inp1.ndim != 0 and inp2.ndim == 0:
            inp2 = reshape(repeat(inp2, prod(Array(inp1.shape))), inp1.shape)

        # both parameters are NDArray
        if inp1.ndim != 0 and inp2.ndim != 0 and inp1.ndim != inp2.ndim:
            # print('reshaping inp2 array')
            new_shape = [1] * (len(inp1.shape) - len(inp2.shape)) + list(inp2.shape)
            inp2 = reshape(inp2, new_shape)

        # first dimension sizes are the same, it can be broadcast together
        if inp1.ndim != 0 and inp2.ndim != 0 and inp1.shape[0] == inp2.shape[0]:
            pass

        # first dimension sizes are not the same and first dimension size of the second input is not 1
        # print(inp1.shape, inp2.shape)
        if inp1.ndim != 0 and inp2.ndim != 0 and inp1.shape[0] != inp2.shape[0] and inp2.shape[0] != 1:
            raise ValueError('cannot broadcast')

        # first dimension sizes are not the same but the first dimension size of the second input is 1, it can be broadcast
        if inp1.ndim != 0 and inp2.ndim != 0 and inp1.shape[0] != inp2.shape[0] and inp2.shape[0] == 1:
            # print('repeating and reshaping inp2 array')
            inp2 = reshape(repeat(inp2, inp1.shape[0]), (inp1.shape[0],) + inp2.shape[1:])

        self.inp1 = inp1
        self.inp2 = inp2

        if self.inp1.ndim != 0 and self.inp2.ndim != 0:
            self.zip = zip(self.inp1, self.inp2)
        else:
            self.zip = None

    def __iter__(self):
        if self.inp1.ndim == 0 or self.inp2.ndim == 0:
            raise ValueError('you have to use get for scalar types')
        return self.zip

    def __next__(self):
        if self.zip is not None:
            return next(self.zip)
        else:
            raise ValueError('you cannot iterate over scalar types')

    def get(self):
        if not (self.inp1.ndim == 0 and self.inp2.ndim == 0):
            raise ValueError('you have to use iter and next for iterables')
        return self.inp1, self.inp2

    def reset(self):
        if self.inp1.ndim != 0 and self.inp2.ndim != 0:
            self.zip = zip(self.inp1, self.inp2)
        else:
            self.zip = None


def _get_type(dtype: DataT):
    if dtype == 'bool':
        return bool
    elif dtype == 'boolean':
        return boolean
    elif dtype == 'int':
        return int
    elif dtype == 'integer':
        return integer
    elif dtype == 'int16':
        return int16
    elif dtype == 'int32':
        return int32
    elif dtype == 'int64':
        return int64
    elif dtype == 'float':
        return float
    elif dtype == 'floating':
        return floating
    elif dtype == 'float16':
        return float16
    elif dtype == 'float32':
        return float32
    elif dtype == 'float64':
        return float64
    else:
        raise ValueError(f'{dtype} is not recognized')


def _normalize_value(value: DataT) -> DataT:
    if isinstance(value, bool):
        value = boolean(value)
    elif isinstance(value, int):
        value = integer(value)
    elif isinstance(value, float):
        value = floating(value)
    elif isinstance(value, (boolean, integer, int16, int32, int64, floating, float16, float32, float64)):
        pass
    else:
        raise ValueError('value type has to be following: bool, boolean, int, integer, int16, int32, int64, float, floating, float16, float32, float64')
    return value


def asarray(arr: ArrayT) -> ArrayT:
    return Array(arr)


def arange(start: float, stop: float, step: float) -> Array:
    return linspace(start, stop, int((stop - start) // step))


def linspace(start: float, stop: float, steps: int) -> Array:
    inc = (stop - start) / steps
    return Array([start + inc * idx for idx in range(steps)])


def eye(rows: int, columns: int) -> Array:
    ret = zeros((rows, columns))
    for row in range(rows):
        if row < columns:
            ret[row][row] = 1
    return ret


def empty(shape) -> Array:
    if len(shape) == 0:
        raise ValueError('array shape has to be at least 1-dimensional')

    ret = []
    for _ in range(shape[0]):
        ret.append(zeros(shape[1:]))
    return Array(ret)


def full(shape) -> Array:
    if len(shape) == 0:
        raise ValueError('array shape has to be at least 1-dimensional')

    ret = []
    for _ in range(shape[0]):
        ret.append(ones(shape[1:]))
    return Array(ret)


def zeros(shape) -> Array:
    if len(shape) == 0:
        raise ValueError('array shape has to be at least 1-dimensional')

    ret = []
    for _ in range(shape[0]):
        ret.append(zeros(shape[1:]))
    return Array(ret)


def ones(shape) -> Array:
    if len(shape) == 0:
        raise ValueError('array shape has to be at least 1-dimensional')

    ret = []
    for _ in range(shape[0]):
        ret.append(ones(shape[1:]))
    return Array(ret)


def ones_like(inp: ArrayT) -> Array:
    return ones(size(inp))


def zeros_like(inp: ArrayT) -> Array:
    return zeros(size(inp))


def concatenate(inputs) -> ArrayT:
    ...


def stack(inputs) -> ArrayT:
    ...


def astype(inp: Union[DataT, ArrayT], dtype: DataT) -> Union[DataT, ArrayT]:
    if isinstance(inp, (bool, boolean, int, integer, int16, int32, int64, float, floating, float16, float32, float64)):
        return _get_type(dtype)(inp)
    elif isinstance(inp, (tuple, list, Array)):
        ret = []
        for data in inp:
            ret.append(astype(data, dtype))
        return type(inp)(ret)
    else:
        raise ValueError('value type is not recognized')


def copy(inp: Union[DataT, ArrayT]) -> Union[DataT, ArrayT]:
    if isinstance(inp, (bool, boolean, int, integer, int16, int32, int64, float, floating, float16, float32, float64)):
        return type(inp)(inp)
    elif isinstance(inp, (tuple, list, Array)):
        return type(inp)(tolist(inp))
    else:
        raise ValueError(f'cannot copy object of type {type(inp)}')


def repeat(inp: Union[DataT, ArrayT], count: DataT, axis=0) -> ArrayT:
    # need to use axis
    ret = []
    for _ in range(count):
        if isinstance(inp, (bool, boolean, int, integer, int16, int32, int64, float, floating, float16, float32, float64)):
            ret.append(inp)
        elif isinstance(inp, (tuple, list, Array)):
            ret.append(tolist(inp))
    return Array(ret)


def split(inp: ArrayT, chunks, axis=0) -> ArrayT:
    # need to use axis
    length = len(inp)
    chunk_length = length // chunks
    if chunk_length * chunks != length:
        raise ValueError(f'array with length {length} cannot be divided into {chunks} chunks')

    ret = []
    for chunk in range(chunks):
        ret.append(inp[chunk * chunk_length: (chunk + 1) * chunk_length])
    return type(inp)(ret)


def tolist(inp: ArrayT) -> list:
    ret = []
    for data in inp:
        if isinstance(data, (bool, int, float)):
            ret.append(data)
        elif isinstance(data, (boolean, integer, int16, int32, int64, floating, float16, float32, float64)):
            ret.append(data.data)
        elif isinstance(data, (tuple, list, Array)):
            ret.append(tolist(data))
        else:
            raise ValueError(f'{type(data)} could not be converted')
    return ret


def getitem(inp: ArrayT, idx: IndexT):
    if isinstance(inp, (tuple, list)):
        if isinstance(idx, int):
            return copy(inp[idx])
        elif isinstance(idx, slice):
            ret = []
            for data in inp[idx]:
                ret.append(data)
            return type(inp)(ret)
        elif isinstance(idx, (tuple, list)) and len(idx) == 1:
            ret = inp[idx[0]]
            if isinstance(ret, (bool, int, float)):
                return ret
            elif isinstance(ret, (tuple, list, Array)):
                return type(inp)(ret)
        elif isinstance(idx, (tuple, list)):
            ret = []
            if isinstance(idx[0], int):
                return getitem(inp[idx[0]], idx[1:])
            elif isinstance(idx[0], slice):
                for data in inp[idx[0]]:
                    ret.append(getitem(data, idx[1:]))
            return type(inp)(ret)
    elif isinstance(inp, Array):
        ret = getitem(tolist(inp), idx)
        if isinstance(ret, (bool, boolean, int, integer, int16, int32, int64, float, floating, float16, float32, float64)):
            return ret
        return Array(ret)
    else:
        raise ValueError(f'object type {type(inp)} is not recognized')


def take_along_axis(inp: Union[DataT, ArrayT], indexes, axis) -> ArrayT:
    ...


def setitem(inp: ArrayT, idx: IndexT, value: Union[DataT, ArrayT]):
    if isinstance(value, (bool, int, float)):
        inp_shape = size(inp)
        inp_size = prod(inp_shape)
        new_value = reshape(repeat(value, inp_size), inp_shape)
    elif isinstance(value, (boolean, integer, int16, int32, int64, floating, float16, float32, float64)):
        inp_shape = size(inp)
        inp_size = prod(inp_shape)
        new_value = reshape(repeat(value, inp_size), inp_shape)
    elif isinstance(value, Array):
        if size(inp) != size(value):
            raise ValueError(f'{size(inp)} and {size(value)} does not match')
        new_value = copy(value)
    else:
        raise ValueError(f'{type(value)} is not recognized as value type')

    if isinstance(inp, (tuple, list)):
        if isinstance(idx, int):
            inp[idx] = new_value[idx]
        elif isinstance(idx, slice):
            for _idx in range(idx.start, idx.stop, idx.step):
                inp[_idx] = new_value[_idx]
        elif isinstance(idx, (tuple, list)) and len(idx) == 1:
            inp[idx[0]] = new_value[idx[0]]
        elif isinstance(idx, (tuple, list)):
            if isinstance(idx[0], int):
                setitem(inp[idx[0]], idx[1:], new_value[idx[0]])
            elif isinstance(idx[0], slice):
                for _idx in range(idx[0].start, idx[0].stop, idx[0].step):
                    setitem(inp[_idx], idx[1:], new_value[_idx])
            else:
                raise ValueError(f'{type(idx[0])} type is not recognized as index')
        else:
            raise ValueError(f'{type(idx)} type is not recognized as index')
    elif isinstance(inp, Array):
        list_inp = tolist(inp)
        setitem(list_inp, idx, new_value)
        return Array(list_inp)
    else:
        raise ValueError(f'object type {type(inp)} is not recognized')


def put_along_axis(inp: Union[DataT, ArrayT], indexes, values, axis) -> ArrayT:
    ...


def where(condition) -> ArrayT:
    ...


def indices(dimensions) -> ArrayT:
    ...


def dim(inp: Union[DataT, ArrayT]) -> int:
    if isinstance(inp, (boolean, int, integer, int16, int32, int64, float, floating, float16, float32, float64)):
        return 0
    return dim(inp[0]) + 1


def size(inp: Union[DataT, ArrayT], axis=None):
    if isinstance(inp, (bool, boolean, int, integer, int16, int32, int64, float, floating, float16, float32, float64)):
        return ()

    if axis is None or axis < 0:
        return tuple([len(inp)] + list(size(inp[0])))
    return tuple((size(inp[0], axis=axis - 1)))


def flatten(inp: ArrayT) -> ArrayT:
    data_type = type(inp)

    ret = []
    for data in inp:
        if dim(data) == 0:
            ret.append(data)
        elif dim(data) == 1:
            ret += tolist(data)
        else:
            ret += tolist(flatten(data))
    return data_type(ret)


def reshape(inp: ArrayT, shape) -> ArrayT:
    flat = flatten(inp)

    subdims = shape[1:]
    subsize = prod(Array(subdims))
    if shape[0] * subsize != len(flat):
        raise ValueError('size does not match or invalid')
    if not subdims:
        return flat
    return Array([reshape(flat[i: i + subsize], subdims) for i in range(0, len(flat), subsize)])


def squeeze(inp: ArrayT, axis=None) -> ArrayT:
    if axis is None:
        if any([x == 1 for x in size(inp)]):
            new_shape = tuple(filter(lambda x: x != 1, size(inp)))
        else:
            new_shape = size(inp)
    else:
        axis_size = size(inp, axis)
        if axis_size != 1:
            raise ValueError('cannot select an axis to squeeze out which has size not equal to one')
        new_shape = list(size(inp))
        new_shape.pop(axis)

    return reshape(inp, new_shape)


def expand_dims(inp: ArrayT, axis) -> ArrayT:
    shape = list(size(inp))
    shape.insert(axis, 1)

    return reshape(inp, shape)


def pad(inp: Union[DataT, ArrayT], padding, mode) -> ArrayT:
    ...


def transpose(inp: Union[DataT, ArrayT], axes) -> ArrayT:
    ...


def fill(inp: ArrayT, value: DataT) -> ArrayT:
    # should use set item and get item
    if isinstance(value, bool):
        value = boolean(value)
    elif isinstance(value, int):
        value = integer(value)
    elif isinstance(value, float):
        value = floating(value)
    elif isinstance(value, (boolean, integer, int16, int32, int64, floating, float16, float32, float64)):
        pass
    else:
        raise ValueError('value type has to be following: bool, boolean, int, integer, int16, int32, int64, float, floating, float16, float32, float64')

    for idx in range(len(inp)):
        data_type = type(inp[idx])
        if isinstance(inp[idx], (bool, int, float)):
            inp[idx] = data_type(copy(value.data))
        elif isinstance(inp[idx], (boolean, integer, int16, int32, int64, floating, float16, float32, float64)):
            inp[idx] = data_type(copy(value))
        else:
            inp[idx] = fill(inp[idx], value)
    return inp


def absolute(inp: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = abs(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = absolute(_inp, out=ret[idx])
    return ret


def negative(inp: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = -inp.data
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = negative(_inp, out=ret[idx])
    return ret


def around(inp: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = round(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = around(_inp, out=ret[idx])
    return ret


def floor(inp: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.floor(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = floor(_inp, out=ret[idx])
    return ret


def ceil(inp: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.ceil(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = ceil(_inp, out=ret[idx])
    return ret


def sqrt(inp: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.sqrt(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = sqrt(_inp, out=ret[idx])
    return ret


def square(inp: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = inp.data ** 2
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = square(_inp, out=ret[idx])
    return ret


def clip(inp: Union[DataT, ArrayT], min_value: DataT, max_value: DataT, *, out=None) -> Union[DataT, ArrayT]:
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        value = inp.data
        if value < min_value:
            value = min_value
        if value > max_value:
            value = max_value
        ret._data = value
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = clip(_inp, min_value, max_value, out=ret[idx])
    return ret


def exp(inp: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.e ** inp.data
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = exp(_inp, out=ret[idx])
    return ret


def tanh(inp: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.tan(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = tanh(_inp, out=ret[idx])
    return ret


def sum(inp: ArrayT, axis=None) -> Union[DataT, ArrayT]:
    if axis is None:
        if inp.ndim == 0:
            return Array(inp.data)

        ret = Array(0)
        for idx in range(len(inp)):
            ret += sum(inp[idx], axis=axis)
        return ret

    # kahan sum
    s = zeros(inp.shape[:axis] + inp.shape[axis + 1:])
    c = zeros(s.shape)
    for i in range(inp.shape[axis]):
        y = inp[(slice(None),) * axis + (i,)] - c
        t = s + y
        c = (t - s) - y
        s = copy(t)
    return s


def mean(inp: ArrayT, axis=None) -> Union[DataT, ArrayT]:
    s = sum(inp, axis)
    n = inp.shape[axis] if axis is not None else prod(Array(inp.shape))
    return s / n


def median(inp: ArrayT, axis=None) -> Union[DataT, ArrayT]:
    ...


def var(inp: ArrayT, axis=None) -> Union[DataT, ArrayT]:
    if axis is not None:
        m_shape = list(inp.shape)
        m_shape[axis] = 1
        m = reshape(mean(inp, axis), m_shape)
        a = absolute(inp - m)
        return mean(a ** 2, axis=axis)

    return mean(absolute(inp - mean(inp)) ** 2)


def std(inp: ArrayT, axis=None) -> Union[DataT, ArrayT]:
    return sqrt(var(inp, axis=axis))


def prod(inp: ArrayT, axis=None) -> int:
    p = 1
    if isinstance(inp, (tuple, list)):
        for data in inp:
            p *= data
    elif isinstance(inp, Array):
        for data in inp.data:
            p *= data.data
    return p


def unique(inp: ArrayT) -> ArrayT:
    ...


def argmax(inp: ArrayT, axis=None) -> Union[DataT, ArrayT]:
    ...


def argmin(inp: ArrayT, axis=None) -> Union[DataT, ArrayT]:
    ...


def amax(inp: ArrayT, axis=None) -> Union[DataT, ArrayT]:
    ...


def amin(inp: ArrayT, axis=None) -> Union[DataT, ArrayT]:
    ...


def add(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    ret = out or zeros(inp1.shape)
    broadcast = Broadcast(inp1, inp2)

    if inp1.ndim == 0 and inp2.ndim == 0:
        inp1, inp2 = broadcast.get()
        ret._data = inp1.data + inp2.data
        return ret

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = add(_inp1, _inp2, out=ret[idx])
    return ret


def sub(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    ret = out or zeros(inp1.shape)
    broadcast = Broadcast(inp1, inp2)

    if inp1.ndim == 0 and inp2.ndim == 0:
        inp1, inp2 = broadcast.get()
        ret._data = inp1.data - inp2.data
        return ret

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = sub(_inp1, _inp2, out=ret[idx])
    return ret


def mul(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    ret = out or zeros(inp1.shape)
    broadcast = Broadcast(inp1, inp2)

    if inp1.ndim == 0 and inp2.ndim == 0:
        inp1, inp2 = broadcast.get()
        ret._data = inp1.data * inp2.data
        return ret

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = mul(_inp1, _inp2, out=ret[idx])
    return ret


def truediv(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    ret = out or zeros(inp1.shape)
    broadcast = Broadcast(inp1, inp2)

    if inp1.ndim == 0 and inp2.ndim == 0:
        inp1, inp2 = broadcast.get()
        ret._data = inp1.data / inp2.data
        return ret

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = truediv(_inp1, _inp2, out=ret[idx])
    return ret


def floordiv(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    ret = out or zeros(inp1.shape)
    broadcast = Broadcast(inp1, inp2)

    if inp1.ndim == 0 and inp2.ndim == 0:
        inp1, inp2 = broadcast.get()
        ret._data = inp1.data // inp2.data
        return ret

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = floor(_inp1, _inp2, out=ret[idx])
    return ret


def power(inp: Union[DataT, ArrayT], p: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = inp.data ** p
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = power(_inp, p, out=ret[idx])
    return ret


def lt(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data < inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = lt(_inp1, _inp2, out=ret[idx])
    return ret


def le(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data <= inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = le(_inp1, _inp2, out=ret[idx])
    return ret


def gt(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data > inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = gt(_inp1, _inp2, out=ret[idx])
    return ret


def ge(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data >= inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = ge(_inp1, _inp2, out=ret[idx])
    return ret


def eq(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data == inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = eq(_inp1, _inp2, out=ret[idx])
    return ret


def ne(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out=None) -> Union[DataT, ArrayT]:
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data != inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = ne(_inp1, _inp2, out=ret[idx])
    return ret


def dot(inp1: ArrayT, inp2: ArrayT, *, out=None) -> ArrayT:
    ...
