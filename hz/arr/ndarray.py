from typing import Union


class NDArray:
    def __init__(self, data=None):
        if isinstance(data, (tuple, list)):
            self._data = []
            for _data in data:
                # check lengths
                self._data.append(NDArray(_data))
        elif isinstance(data, (int, float, bool)):
            self._data = data
        elif isinstance(data, NDArray):
            self._data = data._data
        else:
            raise ValueError(f'data type has to be one of the following (tuple, list, int, float, bool) not {type(data)}')

        self._device = 'cpu'
        self._dtype = 'float32'

    def __repr__(self):
        return str(self)

    def __str__(self):
        if isinstance(self._data, (int, float, bool)):
            return f'{self._data}'

        data = f'{self._data}'

        str_list = [data]
        str_list = list(filter(lambda parameter: parameter != '', str_list))
        string = ', '.join(str_list)

        return f'{string}'

    def __getitem__(self, item):
        if isinstance(self._data, (tuple, list)):
            return self._data[item]
        raise ValueError('you cannot index primitive type')

    def __setitem__(self, key, value):
        if isinstance(self._data, (tuple, list)):
            self._data[key] = value
            return
        raise ValueError('you cannot index primitive type')

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __mul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        return self.div(other)

    def __pow__(self, power, modulo=None):
        return self.power(power)

    def add(self, other, *, out=None):
        ...

    def sub(self, other, *, out=None):
        ...

    def mul(self, other, *, out=None):
        ...

    def div(self, other, *, out=None):
        ...

    def power(self, p, *, out=None):
        ...

    @property
    def device(self):
        return self._device

    @property
    def data(self):
        return self._data

    def size(self, dim=None) -> Union[tuple, int]:
        if dim is None:
            return self.shape
        return self.shape[dim]

    def dim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple:
        if isinstance(self._data, (tuple, list)):
            return tuple([len(self._data)] + list(self._data[0].shape))
        return ()

    @property
    def ndim(self) -> int:
        if isinstance(self._data, (tuple, list)):
            return self._data[0].ndim + 1
        return 0

    @property
    def dtype(self):
        return self._dtype
