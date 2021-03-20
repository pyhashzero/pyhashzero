import math
from copy import deepcopy
from typing import (
    List,
    Tuple,
    Union
)

from hz.arr.array import Array
from hz.arr.data import Data
from hz.arr.shape import Shape
from hz.arr.type import Type

D = Union[int, float, bool, Data]
A = Union[Tuple[D], List[D], Array]
T = Union[str, Type]
S = Union[Tuple[int], List[int], Shape]


class Broadcast:
    def __init__(self, inp1, inp2):
        if not isinstance(inp2, Array):
            inp2 = Array(inp2)

        # both parameters are float, int, bool
        if inp1.ndim == 0 and inp2.ndim == 0:
            pass

        # first parameter is NDArray, second parameter is float, int, bool
        if inp1.ndim != 0 and inp2.ndim == 0:
            inp2 = reshape(repeat(inp2, inp1.shape[0]), (inp1.shape[0],))

        # both parameters are NDArray
        if inp1.ndim != 0 and inp2.ndim != 0 and inp1.ndim != inp2.ndim:
            new_shape = [1] * (len(inp1.shape) - len(inp2.shape)) + list(inp2.shape)
            inp2 = reshape(inp2, new_shape)

        # first dimension sizes are the same, it can be broadcast together
        if inp1.ndim != 0 and inp2.ndim != 0 and inp1.shape[0] == inp2.shape[0]:
            pass

        # first dimension sizes are not the same and first dimension size of the second input is not 1
        if inp1.ndim != 0 and inp2.ndim != 0 and inp1.shape[0] != inp2.shape[0] and inp2.shape[0] != 1:
            raise ValueError('cannot broadcast')

        # first dimension sizes are not the same but the first dimension size of the second input is 1, it can be broadcast
        if inp1.ndim != 0 and inp2.ndim != 0 and inp1.shape[0] != inp2.shape[0] and inp2.shape[0] == 1:
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


def fill(inp, value) -> 'Array':
    if inp.ndim == 0:
        inp._data = value
        return inp

    for idx in range(len(inp)):
        inp[idx] = fill(inp[idx], value)
    return inp


def absolute(inp, *, out=None) -> 'Array':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = abs(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = absolute(_inp, out=ret[idx])
    return ret


def around(inp, *, out=None) -> 'Array':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = round(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = around(_inp, out=ret[idx])
    return ret


def floor(inp, *, out=None) -> 'Array':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.floor(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = floor(_inp, out=ret[idx])
    return ret


def ceil(inp, *, out=None) -> 'Array':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.ceil(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = ceil(_inp, out=ret[idx])
    return ret


def negative(inp, *, out=None) -> 'Array':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = -inp.data
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = negative(_inp, out=ret[idx])
    return ret


def sqrt(inp, *, out=None) -> 'Array':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.sqrt(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = sqrt(_inp, out=ret[idx])
    return ret


def square(inp, *, out=None) -> 'Array':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = inp.data ** 2
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = square(_inp, out=ret[idx])
    return ret


def clip(inp, min_value, max_value, *, out=None) -> 'Array':
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


def exp(inp, *, out=None) -> 'Array':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.e ** inp.data
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = exp(_inp, out=ret[idx])
    return ret


def tanh(inp, *, out=None) -> 'Array':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.tan(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = tanh(_inp, out=ret[idx])
    return ret


def argmax(inp, axis=None) -> 'Array':
    ...


def amax(inp, axis=None) -> 'Array':
    ...


def sum(inp, axis=None) -> 'Array':
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


def mean(inp, axis=None) -> 'Array':
    s = sum(inp, axis)
    n = inp.shape[axis] if axis is not None else prod(Array(inp.shape))
    return s / n


def median(inp, axis=None) -> 'Array':
    ...


def std(inp, axis=None) -> 'Array':
    if axis is not None:
        m_shape = list(inp.shape)
        m_shape[axis] = 1
        m = reshape(mean(inp, axis), m_shape)
        a = absolute(inp - m)
        return sqrt(mean(a ** 2, axis=axis))

    return sqrt(mean(absolute(inp - mean(inp)) ** 2))


def var(inp, axis=None) -> 'Array':
    ...


def prod(inp, axis=None) -> 'Array':
    p = 1
    for data in inp.data:
        p *= data.data
    return Array(p)


def unique(inp) -> 'Array':
    ...


def add(inp1, inp2, *, out=None) -> 'Array':
    if not isinstance(inp2, Array):
        inp2 = Array(inp2)

    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data + inp2.data
        return ret

    if inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        for idx, _inp1 in enumerate(inp1):
            ret[idx] = add(_inp1, inp2, out=ret[idx])
        return ret

    if inp1.shape[0] != inp2.shape[0] and inp2.shape[0] == 1:
        ret = out or zeros(inp1.shape)
        for idx, (_inp1, _inp2) in enumerate(zip(inp1, [inp2] * inp1.shape[0])):
            ret[idx] = add(_inp1, _inp2, out=ret[idx])
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = add(_inp1, _inp2, out=ret[idx])
    return ret


def mul(inp1, inp2, *, out=None) -> 'Array':
    if not isinstance(inp2, Array):
        inp2 = Array(inp2)

    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data * inp2.data
        return ret

    if inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        for idx, _inp1 in enumerate(inp1):
            ret[idx] = mul(_inp1, inp2, out=ret[idx])
        return ret

    if inp1.shape[0] != inp2.shape[0] and inp2.shape[0] == 1:
        ret = out or zeros(inp1.shape)
        for idx, (_inp1, _inp2) in enumerate(zip(inp1, [inp2] * inp1.shape[0])):
            ret[idx] = mul(_inp1, _inp2, out=ret[idx])
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = mul(_inp1, _inp2, out=ret[idx])
    return ret


def div(inp1, inp2, *, out=None) -> 'Array':
    if not isinstance(inp2, Array):
        inp2 = Array(inp2)

    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data / inp2.data
        return ret

    if inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        for idx, _inp1 in enumerate(inp1):
            ret[idx] = div(_inp1, inp2, out=ret[idx])
        return ret

    if inp1.shape[0] != inp2.shape[0] and inp2.shape[0] == 1:
        ret = out or zeros(inp1.shape)
        for idx, (_inp1, _inp2) in enumerate(zip(inp1, [inp2] * inp1.shape[0])):
            ret[idx] = div(_inp1, _inp2, out=ret[idx])
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = div(_inp1, _inp2, out=ret[idx])
    return ret


def sub(inp1, inp2, *, out=None) -> 'Array':
    if not isinstance(inp2, Array):
        inp2 = Array(inp2)

    # both parameters are float, int, bool
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data - inp2.data
        return ret

    # first parameter is NDArray, second parameter is float, int, bool
    if inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        for idx, _inp1 in enumerate(inp1):
            ret[idx] = sub(_inp1, inp2, out=ret[idx])
        return ret

    # both parameters are NDArray
    if inp1.ndim != inp2.ndim:
        # need to add new axis on first axis
        new_shape = [1] * (len(inp1.shape) - len(inp2.shape)) + list(inp2.shape)
        inp2 = reshape(inp2, new_shape)

        # raise ValueError(f'dimension mismatch, {inp1.ndim} - {inp2.ndim}')

    # first dimensiton sizes are the same, it can be broadcasted together
    if inp1.shape[0] == inp2.shape[0]:
        ret = out or zeros(inp1.shape)
        for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
            ret[idx] = sub(_inp1, _inp2, out=ret[idx])
        return ret

    # first dimension sizes are not the same and first dimension size of the second input is not 1
    if inp1.shape[0] != inp2.shape[0] and inp2.shape[0] != 1:
        raise ValueError('cannot broadcast')

    # first dimensiton sizes are not the same but the first dimension size of the second input is 1, it can be broadcasted

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, reshape(repeat(inp2, inp1.shape[0]), (inp1.shape[0],) + inp2.shape[1:]))):
        ret[idx] = sub(_inp1, _inp2, out=ret[idx])
    return ret


def power(inp, p, *, out=None) -> 'Array':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = inp.data ** p
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = power(_inp, p, out=ret[idx])
    return ret


def lt(inp1, inp2, *, out=None) -> 'Array':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data < inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = lt(_inp1, _inp2, out=ret[idx])
    return ret


def le(inp1, inp2, *, out=None) -> 'Array':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data <= inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = le(_inp1, _inp2, out=ret[idx])
    return ret


def gt(inp1, inp2, *, out=None) -> 'Array':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data > inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = gt(_inp1, _inp2, out=ret[idx])
    return ret


def ge(inp1, inp2, *, out=None) -> 'Array':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data >= inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = ge(_inp1, _inp2, out=ret[idx])
    return ret


def eq(inp1, inp2, *, out=None) -> 'Array':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data == inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = eq(_inp1, _inp2, out=ret[idx])
    return ret


def ne(inp1, inp2, *, out=None) -> 'Array':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data != inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = ne(_inp1, _inp2, out=ret[idx])
    return ret


def dot(inp1, inp2, *, out=None) -> 'Array':
    ...


def repeat(inp, count, axis=0) -> 'Array':
    ret = []
    for _ in range(count):
        ret.append(inp.list())
    return Array(ret)


def concatenate(inputs) -> 'Array':
    ...


def stack(inputs) -> 'Array':
    ...


def split(inp, chunks, dim=0) -> 'Array':
    ...


def squeeze(inp, axis) -> 'Array':
    ...


def expand_dims(inp, axis) -> 'Array':
    ...


def reshape(inp, shape) -> 'Array':
    flat = flatten(inp)

    subdims = shape[1:]
    subsize = prod(Array(subdims)).data
    if shape[0] * subsize != len(flat):
        raise ValueError('size does not match or invalid')
    if not subdims:
        return flat
    return Array([reshape(flat[i: i + subsize], subdims) for i in range(0, len(flat), subsize)])


def flatten(inp) -> 'Array':
    if inp.ndim == 0:
        return Array([inp.data])

    ret = []
    for data in inp.data:
        ret += flatten(data).data
    return Array(ret)


def transpose(inp, axes) -> 'Array':
    ...


def pad(inp, padding, mode) -> 'Array':
    ...


def indices(dimensions) -> 'Array':
    ...


def where(condition) -> 'Array':
    ...


def take_along_axis(inp, indexes, axis) -> 'Array':
    ...


def put_along_axis(inp, indexes, values, axis) -> 'Array':
    ...


def arange(start, stop, step) -> 'Array':
    ...


def linspace(start, stop, steps) -> 'Array':
    ...


def eye(rows, columns) -> 'Array':
    ...


def empty(shape) -> 'Array':
    if len(shape) == 0:
        return Array(0)

    ret = []
    for _ in range(shape[0]):
        ret.append(zeros(shape[1:]))
    return Array(ret)


def full(shape) -> 'Array':
    if len(shape) == 0:
        return Array(1)

    ret = []
    for _ in range(shape[0]):
        ret.append(ones(shape[1:]))
    return Array(ret)


def zeros(shape) -> 'Array':
    if len(shape) == 0:
        return Array(0)

    ret = []
    for _ in range(shape[0]):
        ret.append(zeros(shape[1:]))
    return Array(ret)


def ones(shape) -> 'Array':
    if len(shape) == 0:
        return Array(1)

    ret = []
    for _ in range(shape[0]):
        ret.append(ones(shape[1:]))
    return Array(ret)


def ones_like(inp) -> 'Array':
    return ones(inp.shape)


def zeros_like(inp) -> 'Array':
    return zeros(inp.shape)


def copy(inp) -> 'Array':
    return deepcopy(inp)


def asarray(arr) -> 'Array':
    return Array(arr)


def astype(dtype) -> 'Array':
    ...

#
#
# def get_item(inp, indexes) -> 'NDArray':
#     ...
#
#
# def set_item(inp, indexes, values) -> 'NDArray':
#     ...
#
#
# def index_with_booleans(inp, indexes) -> 'NDArray':
#     ...
