from hz.ai.nn.core import Tensor


def _set_grad(tensor, data):
    if not (tensor.requires_grad and not hasattr(tensor, "retains_grad") and not tensor.is_leaf):
        if not tensor.is_leaf:
            return

        if tensor.grad is None:
            tensor.grad = Tensor.from_array(data)
        else:
            tensor.grad._data = tensor.grad.data + data
    tensor.backward(data)


def _create_tensor(*tensors, data, func):
    requires_grad = any(map(lambda x: x.requires_grad, tensors))
    grad_fn = None
    if requires_grad:
        grad_fn = func

    tensor = Tensor.from_array(data=data, requires_grad=requires_grad)
    tensor._grad_fn = grad_fn
    return tensor


def concat(*tensors, axis=0) -> 'Tensor':
    pass


def stack(*tensors, axis=0) -> 'Tensor':
    pass


def chunk(tensor, chunks, dim=0):
    pass


def view(inp, size) -> 'Tensor':
    pass


def index_select(inp, dim, index) -> 'Tensor':
    pass


def zero(inp) -> 'Tensor':
    pass


def one(inp) -> 'Tensor':
    pass


def fill(inp, value) -> 'Tensor':
    pass


def squeeze(inp, axis=None) -> 'Tensor':
    pass


def expand_dim(inp, axis=None) -> 'Tensor':
    pass


def transpose(inp, axes=None) -> 'Tensor':
    pass


def abs(inp) -> 'Tensor':
    pass


def round(inp) -> 'Tensor':
    pass


def floor(inp) -> 'Tensor':
    pass


def ceil(inp) -> 'Tensor':
    pass


def clip(inp, min_val, max_val) -> 'Tensor':
    pass


def negative(inp) -> 'Tensor':
    pass


def sum(inp) -> 'Tensor':
    pass


def mean(inp) -> 'Tensor':
    pass


def std(inp) -> 'Tensor':
    pass


def var(inp) -> 'Tensor':
    pass


def add(inp1, inp2) -> 'Tensor':
    pass


def sub(inp1, inp2) -> 'Tensor':
    pass


def mul(inp1, inp2) -> 'Tensor':
    pass


def div(inp1, inp2) -> 'Tensor':
    pass


def pow(inp, p) -> 'Tensor':
    pass


def clone(inp) -> 'Tensor':
    pass


def detach(inp, inplace=False) -> 'Tensor':
    pass


def arange(start=0, stop=0, step=1, requires_grad=False, dtype='float32') -> 'Tensor':
    pass


def linspace(start, end, steps, requires_grad=False, dtype='float32') -> 'Tensor':
    pass


def normal(loc=0.0, scale=1.0, size=None, requires_grad=False, dtype='float32') -> 'Tensor':
    pass


def uniform(low=-1.0, high=1.0, size=None, requires_grad=False, dtype='float32') -> 'Tensor':
    pass


def rand(size, requires_grad=False, dtype='float32') -> 'Tensor':
    pass


def randint(low=0, high=0, size=None, requires_grad=False, dtype='float32') -> 'Tensor':
    pass


def randn(size, requires_grad=False, dtype='float32') -> 'Tensor':
    pass


def eye(rows, columns=None, requires_grad=False, dtype='float32') -> 'Tensor':
    pass


def empty(size, requires_grad=False, dtype='float32') -> 'Tensor':
    pass


def full(size, fill_value, requires_grad=False, dtype='float32') -> 'Tensor':
    pass


def zeros(size, requires_grad=False, dtype='float32') -> 'Tensor':
    pass


def ones(size, requires_grad=False, dtype='float32') -> 'Tensor':
    pass


def relu(inp) -> 'Tensor':
    pass


def sigmoid(inp) -> 'Tensor':
    pass


def softmax(inp) -> 'Tensor':
    pass


def tanh(inp) -> 'Tensor':
    pass


def dense(inp, weight, bias) -> 'Tensor':
    pass


def conv(inp, weight, bias, stride, padding) -> 'Tensor':
    pass


def dropout(inp, keep_prob) -> 'Tensor':
    pass


def batch_norm(inp, weight, bias, running_mean, running_var, momentum, eps, training) -> 'Tensor':
    pass


def max_pool(inp, kernel_size, stride, padding) -> 'Tensor':
    pass


def avg_pool(inp, kernel_size, stride, padding) -> 'Tensor':
    pass


def rnn_relu(inp, hx, all_weights, bias, num_layers, drop, training):
    pass


def rnn_tanh(inp, hx, all_weights, bias, num_layers, drop, training):
    pass


def lstm(inp, hx, all_weights, bias, num_layers, drop, training):
    pass


def gru(inp, hx, all_weights, bias, num_layers, drop, training):
    pass


def rnn_relu_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    pass


def rnn_tanh_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    pass


def lstm_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    pass


def gru_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    pass


def adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, amsgrad, beta1, beta2, lr, weight_decay, eps):
    pass


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
