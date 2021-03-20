class Data:
    def __init__(self, data=None):
        if isinstance(data, (int, float, bool)):
            self._data = data
        elif isinstance(data, Data):
            self._data = data._data
        else:
            raise ValueError(f'data type has to be one of the following (int, float, bool) not {type(data)}')

        self._device = 'cpu'
        self._dtype = 'float32'

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'{self._data}'

    def __bool__(self):
        return self._data != 0

    def __copy__(self):
        return Data(self._data)

    def __deepcopy__(self, memodict=None):
        return Data(self._data)

    def __add__(self, other):
        if isinstance(other, Data):
            return Data(self._data + other.val)
        else:
            return Data(self._data + other)

    def __iadd__(self, other):
        if isinstance(other, Data):
            self._data = self._data + other.val
        else:
            self._data = self._data + other
        return self

    def __sub__(self, other):
        if isinstance(other, Data):
            return Data(self._data - other.val)
        else:
            return Data(self._data - other)

    def __isub__(self, other):
        if isinstance(other, Data):
            self._data = self._data - other.val
        else:
            self._data = self._data - other
        return self

    def __mul__(self, other):
        if isinstance(other, Data):
            return Data(self._data * other.val)
        else:
            return Data(self._data * other)

    def __imul__(self, other):
        if isinstance(other, Data):
            self._data = self._data * other.val
        else:
            self._data = self._data * other
        return self

    def __truediv__(self, other):
        if isinstance(other, Data):
            return Data(self._data / other.val)
        else:
            return Data(self._data / other)

    def __itruediv__(self, other):
        if isinstance(other, Data):
            self._data = self._data / other.val
        else:
            self._data = self._data / other
        return self

    def __pow__(self, power, modulo=None):
        if isinstance(power, Data):
            return Data(self._data ** power.val)
        else:
            return Data(self._data ** power)

    def __ipow__(self, power):
        if isinstance(power, Data):
            self._data = self._data ** power.val
        else:
            self._data = self._data ** power
        return self

    def __neg__(self):
        return Data(-self._data)

    def __lt__(self, other):
        if isinstance(other, Data):
            return Data(self._data < other.val)
        else:
            return Data(self._data < other)

    def __le__(self, other):
        if isinstance(other, Data):
            return Data(self._data <= other.val)
        else:
            return Data(self._data <= other)

    def __gt__(self, other):
        if isinstance(other, Data):
            return Data(self._data > other.val)
        else:
            return Data(self._data > other)

    def __ge__(self, other):
        if isinstance(other, Data):
            return Data(self._data >= other.val)
        else:
            return Data(self._data >= other)

    def __eq__(self, other):
        if isinstance(other, Data):
            return Data(self._data == other.val)
        else:
            return Data(self._data == other)

    def __ne__(self, other):
        if isinstance(other, Data):
            return Data(self._data != other.val)
        else:
            return Data(self._data != other)

    def copy(self):
        return Data(self._data)

    @property
    def device(self):
        return self._device

    @property
    def val(self):
        return self._data

    @property
    def dtype(self):
        return self._dtype
