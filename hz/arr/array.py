from .data import (
    boolean,
    floating,
    integer
)


class Array:
    def __init__(self, data=None):
        if isinstance(data, (tuple, list, Array)):
            self._data = []
            for _data in data:
                if isinstance(_data, bool):
                    self._data.append(boolean(_data))
                elif isinstance(_data, boolean):
                    self._data.append(_data.copy())
                elif isinstance(_data, int):
                    self._data.append(integer(_data))
                elif isinstance(_data, integer):
                    self._data.append(_data.copy())
                elif isinstance(_data, float):
                    self._data.append(floating(_data))
                elif isinstance(_data, floating):
                    self._data.append(_data.copy())
                else:
                    # instead of creating Array object instance, try using list or tuple
                    # when using __getitem__ Array can be created
                    self._data.append(Array(_data))
        elif isinstance(data, (bool, boolean, int, integer, float, floating)):
            raise ValueError(f'Array type cannot be instantiated from {type(data)} type')
        else:
            raise ValueError(f'data type has to be one of the following (tuple, list, Array) not {type(data)}')

        self._device = 'cpu'
        self._dtype = 'float32'

    def __repr__(self):
        return str(self)

    def __str__(self):
        data = f'{self.data}'
        dtype = f'{self.dtype}'
        shape = f'{self.shape}'

        str_list = [data, dtype, shape]
        str_list = list(filter(lambda parameter: parameter != '', str_list))
        string = ', '.join(str_list)

        return f'{string}'

    def __bool__(self):
        return True

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict=None):
        return self.copy()

    def __iter__(self):
        for data in self.data:
            yield data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.getitem(idx)

    def __setitem__(self, key, value):
        self.setitem(key, value)

    def __abs__(self):
        return self.abs()

    def __neg__(self):
        return self.negative()

    def __round__(self, n=None):
        pass

    def __floor__(self):
        pass

    def __ceil__(self):
        pass

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        return self.add(other, out=self)

    def __sub__(self, other):
        return self.sub(other)

    def __isub__(self, other):
        return self.sub(other, out=self)

    def __mul__(self, other):
        return self.mul(other)

    def __imul__(self, other):
        return self.mul(other, out=self)

    def __truediv__(self, other):
        return self.truediv(other)

    def __itruediv__(self, other):
        return self.truediv(other, out=self)

    def __floordiv__(self, other):
        return self.floordiv(other)

    def __ifloordiv__(self, other):
        return self.floordiv(other, out=self)

    def __pow__(self, power, modulo=None):
        return self.power(power)

    def __ipow__(self, other):
        return self.power(other, out=self)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.le(other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.ge(other)

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def astype(self, dtype):
        ...

    def copy(self):
        ...

    def repeat(self):
        ...

    def split(self):
        ...

    def tolist(self):
        ...

    def getitem(self, idx):
        ...

    def take_along_axis(self):
        ...

    def setitem(self, idx, value):
        ...

    def put_along_axis(self):
        ...

    def where(self):
        ...

    def indices(self):
        ...

    def dim(self):
        ...

    def size(self, dim=None):
        ...

    def flatten(self):
        ...

    def reshape(self, size):
        ...

    def squeeze(self):
        ...

    def expand_dims(self):
        ...

    def pad(self):
        ...

    def transpose(self):
        ...

    def fill(self, value):
        ...

    def abs(self):
        ...

    def negative(self, *, out=None):
        ...

    def round(self, *, out=None):
        ...

    def floor(self, *, out=None):
        ...

    def ceil(self, *, out=None):
        ...

    def sqrt(self, *, out=None):
        ...

    def square(self, *, out=None):
        ...

    def clip(self, min_value, max_value, *, out=None):
        ...

    def exp(self, *, out=None):
        ...

    def tanh(self, *, out=None):
        ...

    def sum(self):
        ...

    def mean(self):
        ...

    def median(self):
        ...

    def var(self):
        ...

    def std(self):
        ...

    def prod(self):
        ...

    def unique(self):
        ...

    def argmax(self):
        ...

    def argmin(self):
        ...

    def amax(self):
        ...

    def amin(self):
        ...

    def add(self, other, *, out=None):
        ...

    def sub(self, other, *, out=None):
        ...

    def mul(self, other, *, out=None):
        ...

    def truediv(self, other, *, out=None):
        ...

    def floordiv(self, other, *, out=None):
        ...

    def power(self, p, *, out=None):
        ...

    def lt(self, other):
        ...

    def le(self, other):
        ...

    def gt(self, other):
        ...

    def ge(self, other):
        ...

    def eq(self, other):
        ...

    def ne(self, other):
        ...

    @property
    def device(self):
        return self._device

    @property
    def data(self):
        return self._data

    @property
    def ndim(self):
        return self.dim()

    @property
    def shape(self):
        return self.size()

    @property
    def dtype(self):
        return self._dtype
