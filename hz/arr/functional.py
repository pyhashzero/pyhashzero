import math

from hz.arr.ndarray import NDArray


def fill(inp, value) -> 'NDArray':
    if inp.ndim == 0:
        inp._data = value
        return inp

    for idx in range(len(inp)):
        inp[idx] = fill(inp[idx], value)
    return inp


def absolute(inp, *, out=None) -> 'NDArray':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = abs(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = absolute(_inp, out=ret[idx])
    return ret


def around(inp, *, out=None) -> 'NDArray':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = round(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = around(_inp, out=ret[idx])
    return ret


def floor(inp, *, out=None) -> 'NDArray':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.floor(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = floor(_inp, out=ret[idx])
    return ret


def ceil(inp, *, out=None) -> 'NDArray':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.ceil(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = ceil(_inp, out=ret[idx])
    return ret


def negative(inp, *, out=None) -> 'NDArray':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = -inp.data
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = negative(_inp, out=ret[idx])
    return ret


def sqrt(inp, *, out=None) -> 'NDArray':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.sqrt(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = sqrt(_inp, out=ret[idx])
    return ret


def square(inp, *, out=None) -> 'NDArray':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = inp.data ** 2
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = square(_inp, out=ret[idx])
    return ret


def clip(inp, min_value, max_value, *, out=None) -> 'NDArray':
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


def exp(inp, *, out=None) -> 'NDArray':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.e ** inp.data
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = exp(_inp, out=ret[idx])
    return ret


def tanh(inp, *, out=None) -> 'NDArray':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = math.tan(inp.data)
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = tanh(_inp, out=ret[idx])
    return ret


def argmax(inp, axis) -> 'NDArray':
    ...


def max(inp) -> 'NDArray':
    ...


def sum(inp) -> 'NDArray':
    ...


def mean(inp) -> 'NDArray':
    ...


def median(inp) -> 'NDArray':
    ...


def std(inp) -> 'NDArray':
    ...


def var(inp) -> 'NDArray':
    ...


def unique(inp) -> 'NDArray':
    ...


def add(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data + inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = add(_inp1, _inp2, out=ret[idx])
    return ret


def mul(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data * inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = mul(_inp1, _inp2, out=ret[idx])
    return ret


def div(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data / inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = div(_inp1, _inp2, out=ret[idx])
    return ret


def sub(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data - inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = sub(_inp1, _inp2, out=ret[idx])
    return ret


def power(inp, p, *, out=None) -> 'NDArray':
    if inp.ndim == 0:
        ret = out or zeros(inp.shape)
        ret._data = inp.data ** p
        return ret

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = power(_inp, p, out=ret[idx])
    return ret


def lt(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data < inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = lt(_inp1, _inp2, out=ret[idx])
    return ret


def le(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data <= inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = le(_inp1, _inp2, out=ret[idx])
    return ret


def gt(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data > inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = gt(_inp1, _inp2, out=ret[idx])
    return ret


def ge(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data >= inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = ge(_inp1, _inp2, out=ret[idx])
    return ret


def eq(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data == inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = eq(_inp1, _inp2, out=ret[idx])
    return ret


def ne(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        ret = out or zeros(inp1.shape)
        ret._data = inp1.data != inp2.data
        return ret

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = ne(_inp1, _inp2, out=ret[idx])
    return ret


def dot(inp1, inp2, *, out=None) -> 'NDArray':
    ...


def concatenate(inputs) -> 'NDArray':
    ...


def stack(inputs) -> 'NDArray':
    ...


def split(inp, chunks, dim=0) -> 'NDArray':
    ...


def squeeze(inp, axis) -> 'NDArray':
    ...


def expand_dims(inp, axis) -> 'NDArray':
    ...


def reshape(inp, shape) -> 'NDArray':
    ...


def transpose(inp, axes) -> 'NDArray':
    ...


def pad(inp, padding, mode) -> 'NDArray':
    ...


def indices(dimensions) -> 'NDArray':
    ...


def where(condition) -> 'NDArray':
    ...


def take_along_axis(inp, indexes, axis) -> 'NDArray':
    ...


def put_along_axis(inp, indexes, values, axis) -> 'NDArray':
    ...


def arange(start, stop, step) -> 'NDArray':
    ...


def linspace(start, stop, steps) -> 'NDArray':
    ...


def eye(rows, columns) -> 'NDArray':
    ...


def empty(shape) -> 'NDArray':
    if len(shape) == 0:
        return NDArray(0)

    ret = []
    for _ in range(shape[0]):
        ret.append(zeros(shape[1:]))
    return NDArray(ret)


def full(shape) -> 'NDArray':
    if len(shape) == 0:
        return NDArray(1)

    ret = []
    for _ in range(shape[0]):
        ret.append(ones(shape[1:]))
    return NDArray(ret)


def zeros(shape) -> 'NDArray':
    if len(shape) == 0:
        return NDArray(0)

    ret = []
    for _ in range(shape[0]):
        ret.append(zeros(shape[1:]))
    return NDArray(ret)


def ones(shape) -> 'NDArray':
    if len(shape) == 0:
        return NDArray(1)

    ret = []
    for _ in range(shape[0]):
        ret.append(ones(shape[1:]))
    return NDArray(ret)


def ones_like(inp) -> 'NDArray':
    return ones(inp.shape)


def zeros_like(inp) -> 'NDArray':
    return zeros(inp.shape)


def copy(inp) -> 'NDArray':
    ...


def asarray(arr) -> 'NDArray':
    return NDArray(arr)


def astype(dtype) -> 'NDArray':
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
