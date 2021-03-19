from hz.arr.ndarray import NDArray

newaxis = None


def add(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        return NDArray(inp1.data + inp2.data)

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = _inp1 + _inp2
    return ret


def mul(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        return NDArray(inp1.data * inp2.data)

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = _inp1 * _inp2
    return ret


def div(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        return NDArray(inp1.data / inp2.data)

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = _inp1 / _inp2
    return ret


def sub(inp1, inp2, *, out=None) -> 'NDArray':
    if inp1.ndim == 0 and inp2.ndim == 0:
        return NDArray(inp1.data - inp2.data)

    ret = out or zeros(inp1.shape)
    for idx, (_inp1, _inp2) in enumerate(zip(inp1, inp2)):
        ret[idx] = _inp1 - _inp2
    return ret


def power(inp, p, *, out=None) -> 'NDArray':
    if inp.ndim == 0:
        return NDArray(inp.data ** p)

    ret = out or zeros(inp.shape)
    for idx, _inp in enumerate(inp):
        ret[idx] = _inp ** p
    return ret


def ones_like(inp) -> 'NDArray':
    ...


def zeros_like(inp) -> 'NDArray':
    ...


def negate(inp, *, out=None) -> 'NDArray':
    ...


def split(inp, chunks, dim=0) -> 'NDArray':
    ...


def reshape(inp, shape) -> 'NDArray':
    ...


def unique(inp) -> 'NDArray':
    ...


def asarray(arr) -> 'NDArray':
    return NDArray(arr)


def astype(dtype) -> 'NDArray':
    ...


def put_along_axis(inp, indexes, values, axis) -> 'NDArray':
    ...


def max(inp) -> 'NDArray':
    ...


def where(condition) -> 'NDArray':
    ...


def square(inp) -> 'NDArray':
    ...


def dot(inp1, inp2, *, out=None) -> 'NDArray':
    ...


def transpose(inp, axes) -> 'NDArray':
    ...


def sum(inp) -> 'NDArray':
    ...


def pad(inp, padding, mode) -> 'NDArray':
    ...


def mean(inp) -> 'NDArray':
    ...


def median(inp) -> 'NDArray':
    ...


def std(inp) -> 'NDArray':
    ...


def var(inp) -> 'NDArray':
    ...


def concatenate(inputs) -> 'NDArray':
    ...


def stack(inputs) -> 'NDArray':
    ...


def take_along_axis(inp, indexes, axis) -> 'NDArray':
    ...


def fill(inp, value) -> 'NDArray':
    ...


def squeeze(inp, axis) -> 'NDArray':
    ...


def expand_dims(inp, axis) -> 'NDArray':
    ...


def absolute(inp) -> 'NDArray':
    ...


def around(inp) -> 'NDArray':
    ...


def floor(inp) -> 'NDArray':
    ...


def ceil(inp) -> 'NDArray':
    ...


def clip(inp, min_value, max_value) -> 'NDArray':
    ...


def negative(inp) -> 'NDArray':
    ...


def arange(start, stop, step) -> 'NDArray':
    ...


def linspace(start, stop, steps) -> 'NDArray':
    ...


def eye(rows, columns) -> 'NDArray':
    ...


def empty(shape) -> 'NDArray':
    ...


def full(shape) -> 'NDArray':
    ...


def zeros(shape) -> 'NDArray':
    if len(shape) == 0:
        return NDArray(0)

    ret = []
    for _ in range(shape[0]):
        ret.append(zeros(shape[1:]))
    return NDArray(ret)


def ones(shape) -> 'NDArray':
    ...


def copy(inp) -> 'NDArray':
    ...


def exp(inp) -> 'NDArray':
    ...


def tanh(inp) -> 'NDArray':
    ...


def sqrt(inp) -> 'NDArray':
    ...


def argmax(inp, axis) -> 'NDArray':
    ...


def indices(dimensions) -> 'NDArray':
    ...


def get_item(inp, indexes) -> 'NDArray':
    ...


def set_item(inp, indexes, values) -> 'NDArray':
    ...


def index_with_booleans(inp, indexes) -> 'NDArray':
    ...


def gt(inp1, inp2) -> 'NDArray':
    ...


def lt(inp1, inp2) -> 'NDArray':
    ...


def eq(inp1, inp2) -> 'NDArray':
    ...


def neq(inp1, inp2) -> 'NDArray':
    ...


def gte(inp1, inp2) -> 'NDArray':
    ...


def lte(inp1, inp2) -> 'NDArray':
    ...
