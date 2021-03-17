from hz.arr.ndarray import NDArray


def is_ndarray(obj: object) -> bool:
    return isinstance(obj, NDArray)


def concat(*ndarrays, axis=0) -> 'NDArray':
    pass


def stack(*ndarrays, axis=0) -> 'NDArray':
    pass


def chunk(ndarray, chunks, dim=0):
    pass


def view(inp, size=None) -> 'NDArray':
    pass


def index_select(inp, dim, index) -> 'NDArray':
    pass


def zero(inp) -> 'NDArray':
    pass


def one(inp) -> 'NDArray':
    pass


def fill(inp, value) -> 'NDArray':
    pass


def squeeze(inp, axis=None) -> 'NDArray':
    pass


def expand_dim(inp, axis=None) -> 'NDArray':
    pass


def transpose(inp, axes=None) -> 'NDArray':
    pass


def abs(inp) -> 'NDArray':
    pass


def round(inp) -> 'NDArray':
    pass


def floor(inp) -> 'NDArray':
    pass


def ceil(inp) -> 'NDArray':
    pass


def clip(inp, min_val, max_val) -> 'NDArray':
    pass


def negative(inp) -> 'NDArray':
    pass


def sum(inp) -> 'NDArray':
    pass


def mean(inp) -> 'NDArray':
    pass


def std(inp) -> 'NDArray':
    pass


def var(inp) -> 'NDArray':
    pass


def add(inp1, inp2) -> 'NDArray':
    pass


def sub(inp1, inp2) -> 'NDArray':
    pass


def mul(inp1, inp2) -> 'NDArray':
    pass


def div(inp1, inp2) -> 'NDArray':
    pass


def pow(inp, p) -> 'NDArray':
    pass


def clone(inp) -> 'NDArray':
    pass


def arange(start=0, stop=0, step=1, device='cpu', dtype='float32') -> 'NDArray':
    pass


def linspace(start, end, steps, device='cpu', dtype='float32') -> 'NDArray':
    pass


def normal(loc=0.0, scale=1.0, size=None, device='cpu', dtype='float32') -> 'NDArray':
    pass


def uniform(low=-1.0, high=1.0, size=None, device='cpu', dtype='float32') -> 'NDArray':
    pass


def rand(size, device='cpu', dtype='float32') -> 'NDArray':
    pass


def randint(low=0, high=0, size=None, device='cpu', dtype='float32') -> 'NDArray':
    pass


def randn(size, device='cpu', dtype='float32') -> 'NDArray':
    pass


def eye(rows, columns=None, device='cpu', dtype='float32') -> 'NDArray':
    pass


def empty(size, device='cpu', dtype='float32') -> 'NDArray':
    pass


def full(size, fill_value, device='cpu', dtype='float32') -> 'NDArray':
    pass


def zeros(size, device='cpu', dtype='float32') -> 'NDArray':
    pass


def ones(size, device='cpu', dtype='float32') -> 'NDArray':
    pass


def normal_like(ndarray, loc=0.0, scale=1.0, device='cpu', dtype='float32') -> 'NDArray':
    return normal(loc, scale, ndarray.shape, device, dtype)


def uniform_like(ndarray, low=-1.0, high=1.0, device='cpu', dtype='float32') -> 'NDArray':
    return uniform(low, high, ndarray.shape, device, dtype)


def rand_like(ndarray, device='cpu', dtype='float32') -> 'NDArray':
    return rand(ndarray.shape, device, dtype)


def randint_like(ndarray, low=0, high=0, device='cpu', dtype='float32') -> 'NDArray':
    return randint(low, high, ndarray.shape, device, dtype)


def randn_like(ndarray, device='cpu', dtype='float32') -> 'NDArray':
    return randn(ndarray.shape, device, dtype)


def eye_like(ndarray, device='cpu', dtype='float32') -> 'NDArray':
    return eye(ndarray.shape[0], ndarray.shape[1], device, dtype)


def empty_like(ndarray, device='cpu', dtype='float32') -> 'NDArray':
    return empty(ndarray.shapeh, device, dtype)


def full_like(ndarray, fill_value, device='cpu', dtype='float32') -> 'NDArray':
    return full(ndarray.shape, fill_value, device, dtype)


def zeros_like(ndarray, device='cpu', dtype='float32') -> 'NDArray':
    return zeros(ndarray.shape, device, dtype)


def ones_like(ndarray, device='cpu', dtype='float32') -> 'NDArray':
    return ones(ndarray.shape, device, dtype)


def from_array(data, dtype='float32') -> 'NDArray':
    return NDArray(data)


def to_array(inp):
    if inp.device != 'cpu':
        raise TypeError('can\'t convert cuda:0 device type ndarray to numpy. Use NDArray.cpu() to copy the ndarray to host memory first.')
    return inp.data


def half(inp) -> 'NDArray':
    pass


def single(inp) -> 'NDArray':
    pass


def double(inp) -> 'NDArray':
    pass


def cpu(inp) -> 'NDArray':
    if inp.device == 'cpu':
        return inp

    inp._device = 'cpu'
    return inp


def gpu(inp) -> 'NDArray':
    if inp.device == 'gpu':
        return inp

    inp._device = 'gpu'
    return inp
