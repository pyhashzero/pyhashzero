from typing import Union


class Broadcast(object):
    index = property(lambda self: object(), lambda self, v: None, lambda self: None)
    iters = property(lambda self: object(), lambda self, v: None, lambda self: None)
    nd = property(lambda self: object(), lambda self, v: None, lambda self: None)
    ndim = property(lambda self: object(), lambda self, v: None, lambda self: None)
    numiter = property(lambda self: object(), lambda self, v: None, lambda self: None)
    shape = property(lambda self: object(), lambda self, v: None, lambda self: None)
    size = property(lambda self: object(), lambda self, v: None, lambda self: None)


class DType:
    pass


class NDArray:
    def __init__(self, data=None):
        self._data = data
        self._device = 'cpu'

    def __repr__(self):
        return str(self)

    def __str__(self):
        data = f'data={self._data}'
        device = f'device={self.device}'

        str_list = [device, data]
        str_list = list(filter(lambda parameter: parameter != '', str_list))
        string = ', '.join(str_list)

        return f'NDArray({string})'

    def __getitem__(self, item):
        data = self.data[item]
        ndarray = NDArray.from_array(data)
        if self.device == 'gpu':
            ndarray = ndarray.gpu()
        return ndarray

    def __setitem__(self, key, value):
        self._data[key] = value.data

    def __add__(self, other) -> 'NDArray':
        return self.add(other)

    def __sub__(self, other) -> 'NDArray':
        return self.sub(other)

    def __mul__(self, other) -> 'NDArray':
        return self.mul(other)

    def __truediv__(self, other) -> 'NDArray':
        return self.div(other)

    def __pow__(self, power, modulo=None) -> 'NDArray':
        return self.pow(power)

    def chunk(self, chunks, dim=0):
        ...

    def view(self, size) -> 'NDArray':
        ...

    def index_select(self, dim, index) -> 'NDArray':
        ...

    def zero(self) -> 'NDArray':
        ...

    def one(self) -> 'NDArray':
        ...

    def fill(self, value) -> 'NDArray':
        ...

    def squeeze(self, axis=None) -> 'NDArray':
        ...

    def expand_dim(self, axis=None) -> 'NDArray':
        ...

    def transpose(self, axes) -> 'NDArray':
        ...

    def abs(self) -> 'NDArray':
        ...

    def round(self) -> 'NDArray':
        ...

    def floor(self) -> 'NDArray':
        ...

    def ceil(self) -> 'NDArray':
        ...

    def clip(self, min_val, max_val) -> 'NDArray':
        ...

    def negative(self) -> 'NDArray':
        ...

    def sum(self) -> 'NDArray':
        ...

    def mean(self) -> 'NDArray':
        ...

    def std(self) -> 'NDArray':
        ...

    def var(self) -> 'NDArray':
        ...

    def add(self, other) -> 'NDArray':
        ...

    def sub(self, other) -> 'NDArray':
        ...

    def mul(self, other) -> 'NDArray':
        ...

    def div(self, other) -> 'NDArray':
        ...

    def pow(self, power) -> 'NDArray':
        ...

    def clone(self) -> 'NDArray':
        ...

    def detach(self, inplace=False) -> 'NDArray':
        ...

    @staticmethod
    def from_array(data, requires_grad=False) -> 'NDArray':
        ...

    def to_array(self):
        ...

    def half(self) -> 'NDArray':
        ...

    def single(self) -> 'NDArray':
        ...

    def double(self) -> 'NDArray':
        ...

    def cpu(self) -> 'NDArray':
        ...

    def gpu(self) -> 'NDArray':
        ...

    def size(self, dim=None) -> Union[tuple, int]:
        if dim is None:
            return self._data.shape
        return self._data.shape[dim]

    def dim(self) -> int:
        return len(self._data.shape)

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return len(self._data.shape)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def device(self) -> str:
        return self._device

    @property
    def data(self):
        return self._data
