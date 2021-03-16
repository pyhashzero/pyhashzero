from hz.ai.nn.core import Tensor


# need to implement inplace

# should have c / c++ codes to use them in functional apis

def _check_tensor_data_types(*tensors):
    return True
    print([tensor.dtype for tensor in tensors])
    iterator = iter(tensors)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first.dtype == x.dtype for x in iterator)


def _check_tensor_data_shapes(*tensors):
    return True
    print([tensor.shape for tensor in tensors])
    iterator = iter(tensors)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first.shape == x.shape for x in iterator)


def _check_tensor_devices(*tensors):
    iterator = iter(tensors)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first.device == x.device for x in iterator)


def _check_tensors(*tensors):
    if not _check_tensor_data_types(*tensors):
        raise ValueError('data types are not matching')
    if not _check_tensor_data_shapes(*tensors):
        raise ValueError('data shapes are not matching')
    if not _check_tensor_devices(*tensors):
        raise ValueError('devices are not matching')

    if len(tensors) == 0:
        raise ValueError('there should be at least one tensor')

    if tensors[0].device not in ('cpu', 'gpu'):
        raise ValueError('device has to be either \'cpu\' or \'gpu\'')


def _get_engine(*tensors):
    if (isinstance(tensors[0], Tensor) and tensors[0].device == 'gpu') or (isinstance(tensors[0], str) and tensors[0] == 'gpu'):
        import H0.nn.engine.cupy as engine
        return engine

    import hz.ai.nn.engine.numpy as engine
    return engine


def _set_grad(tensor, data):
    if not (tensor.requires_grad and not hasattr(tensor, "retains_grad") and not tensor.is_leaf):
        if not tensor.is_leaf:
            return

        if tensor.grad is None:
            tensor.grad = from_array(data)
            if tensor.device == 'gpu':
                tensor.grad.gpu()
        else:
            tensor.grad._data = tensor.grad.data + data


def _create_tensor(*tensors, data, func):
    requires_grad = any(map(lambda x: x.requires_grad, tensors))
    grad_fn = None
    if requires_grad:
        grad_fn = func

    tensor = from_array(data=data, requires_grad=requires_grad)
    tensor._grad_fn = grad_fn
    if tensors[0].device == 'gpu':
        tensor.gpu()
    return tensor


def is_tensor(obj: object) -> bool:
    return isinstance(obj, Tensor)


def concat(*tensors, axis=0) -> 'Tensor':
    _check_tensors(*tensors)
    engine = _get_engine(*tensors)
    return engine.concat(*tensors, axis=axis)


def stack(*tensors, axis=0) -> 'Tensor':
    _check_tensors(*tensors)
    engine = _get_engine(*tensors)
    return engine.stack(*tensors, axis=axis)


def chunk(tensor, chunks, dim=0):
    _check_tensors(tensor)
    engine = _get_engine(tensor)
    return engine.chunk(tensor, chunks, dim)


def view(inp, size=None) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.view(inp, size)


def index_select(inp, dim, index) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.index_select(inp, dim, index)


def zero(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.zero(inp)


def one(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.one(inp)


def fill(inp, value) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.fill(inp, value=value)


def squeeze(inp, axis=None) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.squeeze(inp, axis=axis)


def expand_dim(inp, axis=None) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.expand_dim(inp, axis=axis)


def transpose(inp, axes=None) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.transpose(inp, axes=axes)


def abs(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.abs(inp)


def round(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.round(inp)


def floor(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.floor(inp)


def ceil(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.ceil(inp)


def clip(inp, min_val, max_val) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.clip(inp, min_val=min_val, max_val=max_val)


def negative(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.negative(inp)


def sum(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.sum(inp)


def mean(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.mean(inp)


def std(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.std(inp)


def var(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.var(inp)


def add(inp1, inp2) -> 'Tensor':
    _check_tensors(inp1, inp2)
    engine = _get_engine(inp1, inp2)
    return engine.add(inp1, inp2)


def sub(inp1, inp2) -> 'Tensor':
    _check_tensors(inp1, inp2)
    engine = _get_engine(inp1, inp2)
    return engine.sub(inp1, inp2)


def mul(inp1, inp2) -> 'Tensor':
    _check_tensors(inp1, inp2)
    engine = _get_engine(inp1, inp2)
    return engine.mul(inp1, inp2)


def div(inp1, inp2) -> 'Tensor':
    if not isinstance(inp2, Tensor):
        inp2 = Tensor.from_array(inp2)

    _check_tensors(inp1, inp2)
    engine = _get_engine(inp1, inp2)
    return engine.div(inp1, inp2)


def pow(inp, p) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.pow(inp, p)


def clone(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.clone(inp)


def detach(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.detach(inp)


def arange(start=0, stop=0, step=1, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return engine.arange(start, stop, step, requires_grad, dtype)


def linspace(start, end, steps, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return engine.linspace(start, end, steps, requires_grad, dtype)


def normal(loc=0.0, scale=1.0, size=None, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return engine.normal(loc, scale, size, requires_grad, dtype)


def uniform(low=-1.0, high=1.0, size=None, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return engine.uniform(low, high, size, requires_grad, dtype)


def rand(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return engine.rand(size, requires_grad, dtype)


def randint(low=0, high=0, size=None, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return engine.randint(low, high, *size, requires_grad, dtype)


def randn(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return engine.randn(size, requires_grad, dtype)


def eye(rows, columns=None, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return engine.eye(rows, columns, requires_grad, dtype)


def empty(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return engine.empty(size, requires_grad, dtype)


def full(size, fill_value, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return engine.full(size, fill_value, requires_grad, dtype)


def zeros(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return engine.zeros(size, requires_grad, dtype)


def ones(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return engine.ones(size, requires_grad, dtype)


def normal_like(tensor, loc=0.0, scale=1.0, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return normal(loc, scale, tensor.shape, requires_grad, device, dtype)


def uniform_like(tensor, low=-1.0, high=1.0, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return uniform(low, high, tensor.shape, requires_grad, device, dtype)


def rand_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return rand(tensor.shape, requires_grad, device, dtype)


def randint_like(tensor, low=0, high=0, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return randint(low, high, tensor.shape, requires_grad, device, dtype)


def randn_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return randn(tensor.shape, requires_grad, device, dtype)


def eye_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return eye(tensor.shape[0], tensor.shape[1], requires_grad, device, dtype)


def empty_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return empty(tensor.shapeh, requires_grad, device, dtype)


def full_like(tensor, fill_value, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return full(tensor.shape, fill_value, requires_grad, device, dtype)


def zeros_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return zeros(tensor.shape, requires_grad, device, dtype)


def ones_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return ones(tensor.shape, requires_grad, device, dtype)


def from_array(data, requires_grad=False, dtype='float32') -> 'Tensor':
    import numpy as np
    return Tensor(np.copy(data).astype(dtype), requires_grad=requires_grad)


def to_array(inp):
    _check_tensors(inp)

    if inp.device != 'cpu':
        raise TypeError('can\'t convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.')
    if inp.requires_grad:
        raise RuntimeError('Can\'t call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.')
    import numpy as np
    return np.array(inp.data, copy=True)


def half(inp) -> 'Tensor':
    inp._data = inp.data.astype('float16')
    return inp


def single(inp) -> 'Tensor':
    inp._data = inp.data.astype('float32')
    return inp


def double(inp) -> 'Tensor':
    inp._data = inp.data.astype('float64')
    return inp


def cpu(inp) -> 'Tensor':
    if inp.device == 'cpu':
        return inp

    import cupy as cp
    inp._device = 'cpu'
    inp._data = cp.asnumpy(inp.data)
    return inp


def gpu(inp) -> 'Tensor':
    if inp.device == 'gpu':
        return inp

    import cupy as cp
    inp._device = 'gpu'
    inp._data = cp.array(inp.data)
    return inp


def relu(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.relu(inp)


def sigmoid(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.sigmoid(inp)


def softmax(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.softmax(inp)


def tanh(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.tanh(inp)


def dense(inp, weight, bias) -> 'Tensor':
    _check_tensors(inp, weight, bias)
    engine = _get_engine(inp, weight, bias)
    return engine.dense(inp, weight, bias)


def conv(inp, weight, bias, stride, padding) -> 'Tensor':
    _check_tensors(inp, weight, bias)
    engine = _get_engine(inp, weight, bias)
    return engine.conv(inp, weight, bias, stride, padding)


def dropout(inp, keep_prob) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.dropout(inp, keep_prob)


def batch_norm(inp, weight, bias, running_mean, running_var, momentum, eps, training) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.batch_norm(inp, weight, bias, running_mean, running_var, momentum, eps, training)


def max_pool(inp, kernel_size, stride, padding) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.max_pool(inp, kernel_size, stride, padding)


def avg_pool(inp, kernel_size, stride, padding) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.avg_pool(inp, kernel_size, stride, padding)


def rnn_relu(inp, hx, all_weights, bias, num_layers, drop, training):
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.rnn_relu(inp, hx, all_weights, bias, num_layers, drop, training)


def rnn_tanh(inp, hx, all_weights, bias, num_layers, drop, training):
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.rnn_tanh(inp, hx, all_weights, bias, num_layers, drop, training)


def lstm(inp, hx, all_weights, bias, num_layers, drop, training):
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.lstm(inp, hx, all_weights, bias, num_layers, drop, training)


def gru(inp, hx, all_weights, bias, num_layers, drop, training):
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.gru(inp, hx, all_weights, bias, num_layers, drop, training)


def rnn_relu_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.rnn_relu_cell(inp, h, w_ih, w_hh, b_ih, b_hh)


def rnn_tanh_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    _check_tensors(inp)
    engine = _get_engine(inp)
    return engine.rnn_tanh_cell(inp, h, w_ih, w_hh, b_ih, b_hh)


def lstm_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    def lstm_cell_backward(gradient):
        inp.backward()
        h.backward()
        w_ih.backward()
        w_hh.backward()
        if b_ih is not None:
            b_ih.backward()
        if b_hh is not None:
            b_hh.backward()

    hx, cx = h
    gates = dense(inp, w_ih, b_ih) + dense(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = sigmoid(ingate)
    forgetgate = sigmoid(forgetgate)
    cellgate = tanh(cellgate)
    outgate = sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * tanh(cy)

    return hy, cy


def gru_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    def gru_cell_backward(gradient):
        inp.backward()
        h.backward()
        w_ih.backward()
        w_hh.backward()
        if b_ih is not None:
            b_ih.backward()
        if b_hh is not None:
            b_hh.backward()

    gi = dense(inp, w_ih, b_ih)
    gh = dense(h, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = sigmoid(i_r + h_r)
    inputgate = sigmoid(i_i + h_i)
    newgate = tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (h - newgate)

    return hy


def adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, amsgrad, beta1, beta2, lr, weight_decay, eps):
    _check_tensors(*params)
    engine = _get_engine(*params)
    return engine.adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, amsgrad, beta1, beta2, lr, weight_decay, eps)


class _FunctionBase(object):
    @classmethod
    def apply(cls, *args, **kwargs):
        pass

    def name(self, *args, **kwargs):
        pass

    def register_hook(self, *args, **kwargs):
        pass

    def _do_backward(self, gradient):
        self.fn(gradient)

    def _register_hook_dict(self, tensor):
        self.tensor = tensor

    def __init__(self, fn):
        self.fn = fn

    dirty_tensors = property(lambda self: object(), lambda self, v: None, lambda self: None)
    materialize_grads = property(lambda self: object(), lambda self, v: None, lambda self: None)
    metadata = property(lambda self: object(), lambda self, v: None, lambda self: None)
    needs_input_grad = property(lambda self: object(), lambda self, v: None, lambda self: None)
    next_functions = property(lambda self: object(), lambda self, v: None, lambda self: None)
    non_differentiable = property(lambda self: object(), lambda self, v: None, lambda self: None)
    requires_grad = property(lambda self: object(), lambda self, v: None, lambda self: None)
    saved_tensors = property(lambda self: object(), lambda self, v: None, lambda self: None)
    saved_variables = property(lambda self: object(), lambda self, v: None, lambda self: None)
    to_save = property(lambda self: object(), lambda self, v: None, lambda self: None)
