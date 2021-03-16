import math

import numpy as np

from hz.ai.nn.core import Tensor
from hz.ai.nn.utility import _calculate_output_dims


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
    def concat_backward(gradient):
        grad_arrays = np.split(gradient.data, len(tensors), axis=axis)
        for idx, tensor in enumerate(tensors):
            _set_grad(tensor, data=grad_arrays[idx] * np.ones_like(tensor.data))

    return _create_tensor(*tensors, data=np.concatenate(list(map(lambda x: x.data, tensors)), axis=axis), func=concat_backward)


def stack(*tensors, axis=0) -> 'Tensor':
    def stack_backward(gradient):
        grad_arrays = np.split(gradient.data, len(tensors), axis=axis)
        for idx, tensor in enumerate(tensors):
            _set_grad(tensor, data=grad_arrays[idx] * np.ones_like(tensor.data))

    return _create_tensor(*tensors, data=np.stack(list(map(lambda x: x.data, tensors)), axis=axis), func=stack_backward)


def chunk(tensor, chunks, dim=0):
    def chunk_backward(gradient):
        _set_grad(tensor, gradient.data * np.ones_like(tensor.data) / chunks)

    arrays = np.split(tensor.data, chunks, dim)

    tensors = []
    for array in arrays:
        tensors.append(_create_tensor(tensor, data=array, func=chunk_backward))
    return tensors


def view(inp, size) -> 'Tensor':
    def view_backward(gradient):
        _set_grad(inp, gradient.data.reshape(inp.shape))

    return _create_tensor(inp, data=inp.data.reshape(size), func=view_backward)


def index_select(inp, dim, index) -> 'Tensor':
    def index_select_backward(gradient):
        unique, counts = np.unique(index.data.astype('int'), return_counts=True)
        count_dict = dict(zip(unique, counts))

        index_array = np.asarray([val for val in range(inp.size(dim))]).astype('int')
        count_array = np.asarray([count_dict.get(val, 0) for val in range(inp.size(dim))])

        grad_array = np.zeros_like(gradient.data)
        np.put_along_axis(grad_array, index_array, count_array, axis=dim)
        _set_grad(inp, data=grad_array)

    return _create_tensor(inp, data=np.take_along_axis(inp.data, index.data.astype('int'), dim), func=index_select_backward)


def zero(inp) -> 'Tensor':
    inp._data = np.zeros_like(inp.data)
    return inp


def one(inp) -> 'Tensor':
    inp._data = np.ones_like(inp.data)
    return inp


def fill(inp, value) -> 'Tensor':
    inp._data.fill(value)
    return inp


def squeeze(inp, axis=None) -> 'Tensor':
    def squeeze_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=np.squeeze(inp.data, axis=axis), func=squeeze_backward)


def expand_dim(inp, axis=None) -> 'Tensor':
    def expand_dim_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=np.expand_dims(inp.data, axis=axis), func=expand_dim_backward)


def transpose(inp, axes=None) -> 'Tensor':
    def transpose_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=np.transpose(inp.data, axes=axes), func=transpose_backward)


def abs(inp) -> 'Tensor':
    def abs_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=np.abs(inp.data), func=abs_backward)


def round(inp) -> 'Tensor':
    def round_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=np.around(inp.data), func=round_backward)


def floor(inp) -> 'Tensor':
    def floor_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=np.floor(inp.data), func=floor_backward)


def ceil(inp) -> 'Tensor':
    def ceil_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=np.ceil(inp.data), func=ceil_backward)


def clip(inp, min_val, max_val) -> 'Tensor':
    def clip_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=np.clip(inp.data, min_val, max_val), func=clip_backward)


def negative(inp) -> 'Tensor':
    def negative_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=np.negative(inp.data), func=negative_backward)


def sum(inp) -> 'Tensor':
    def summation_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=np.sum(inp.data), func=summation_backward)


def mean(inp) -> 'Tensor':
    def mean_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=np.mean(inp.data), func=mean_backward)


def std(inp) -> 'Tensor':
    def standard_deviation_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=np.std(inp.data), func=standard_deviation_backward)


def var(inp) -> 'Tensor':
    def variance_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=np.var(inp.data), func=variance_backward)


def add(inp1, inp2) -> 'Tensor':
    def add_backward(gradient):
        _set_grad(inp1, gradient.data * np.ones_like(inp1.data))
        _set_grad(inp2, gradient.data * np.ones_like(inp2.data))

    return _create_tensor(inp1, inp2, data=inp1.data + inp2.data, func=add_backward)


def sub(inp1, inp2) -> 'Tensor':
    def sub_backward(gradient):
        _set_grad(inp1, gradient.data * np.ones_like(inp1.data))
        _set_grad(inp2, gradient.data * -np.ones_like(inp2.data))

    return _create_tensor(inp1, inp2, data=inp1.data - inp2.data, func=sub_backward)


def mul(inp1, inp2) -> 'Tensor':
    def mul_backward(gradient):
        _set_grad(inp1, gradient.data * inp2.data)
        _set_grad(inp2, gradient.data * inp1.data)

    return _create_tensor(inp1, inp2, data=inp1.data * inp2.data, func=mul_backward)


def div(inp1, inp2) -> 'Tensor':
    def div_backward(gradient):
        _set_grad(inp1, gradient.data * (1 / inp2.data))
        _set_grad(inp2, gradient.data * inp1.data)

    return _create_tensor(inp1, inp2, data=inp1.data / inp2.data, func=div_backward)


def pow(inp, p) -> 'Tensor':
    def power_backward(gradient):
        _set_grad(inp, gradient.data * p * (inp.data ** (p - 1)))

    return _create_tensor(inp, data=inp.data ** p, func=power_backward)


def clone(inp) -> 'Tensor':
    def clone_backward(gradient):
        _set_grad(inp, gradient.data * np.ones_like(inp.data))

    return _create_tensor(inp, data=inp.data, func=clone_backward)


def detach(inp, inplace=False) -> 'Tensor':
    if inplace:
        inp._grad_fn = None
        inp._requires_grad = False
        return inp

    _clone = inp.clone()
    _clone._grad_fn = None
    _clone._requires_grad = False
    return _clone


def arange(start=0, stop=0, step=1, requires_grad=False, dtype='float32') -> 'Tensor':
    return Tensor.from_array(np.arange(start, stop, step).astype(dtype), requires_grad)


def linspace(start, end, steps, requires_grad=False, dtype='float32') -> 'Tensor':
    return Tensor.from_array(np.linspace(start, end, steps).astype(dtype), requires_grad)


def normal(loc=0.0, scale=1.0, size=None, requires_grad=False, dtype='float32') -> 'Tensor':
    return Tensor.from_array(np.random.normal(loc, scale, size).astype(dtype), requires_grad)


def uniform(low=-1.0, high=1.0, size=None, requires_grad=False, dtype='float32') -> 'Tensor':
    return Tensor.from_array(np.random.uniform(low, high, size).astype(dtype), requires_grad)


def rand(size, requires_grad=False, dtype='float32') -> 'Tensor':
    return Tensor.from_array(np.random.rand(size).astype(dtype), requires_grad)


def randint(low=0, high=0, size=None, requires_grad=False, dtype='float32') -> 'Tensor':
    return Tensor.from_array(np.random.randint(low, high, *size).astype(dtype), requires_grad)


def randn(size, requires_grad=False, dtype='float32') -> 'Tensor':
    return Tensor.from_array(np.random.randn(size).astype(dtype), requires_grad)


def eye(rows, columns=None, requires_grad=False, dtype='float32') -> 'Tensor':
    return Tensor.from_array(np.eye(rows, columns).astype(dtype), requires_grad)


def empty(size, requires_grad=False, dtype='float32') -> 'Tensor':
    return Tensor.from_array(np.empty(size).astype(dtype), requires_grad)


def full(size, fill_value, requires_grad=False, dtype='float32') -> 'Tensor':
    return Tensor.from_array(np.full(size, fill_value).astype(dtype), requires_grad)


def zeros(size, requires_grad=False, dtype='float32') -> 'Tensor':
    return Tensor.from_array(np.zeros(size).astype(dtype), requires_grad)


def ones(size, requires_grad=False, dtype='float32') -> 'Tensor':
    return Tensor.from_array(np.ones(size).astype(dtype), requires_grad)


def relu(inp) -> 'Tensor':
    def relu_backward(gradient):
        out = np.zeros_like(inp.data)
        out[inp.data <= 0] = 0
        out[inp.data > 0] = 1

        _set_grad(inp, gradient.data * out)

    arr = inp.data
    arr[arr <= 0] = 0
    return _create_tensor(inp, data=arr, func=relu_backward)


def sigmoid(inp) -> 'Tensor':
    def sigmoid_backward(gradient):
        _set_grad(inp, gradient.data * output_array * (1 - output_array))

    output_array = 1 / (1 + np.exp(-inp.data))
    return _create_tensor(inp, data=output_array, func=sigmoid_backward)


def softmax(inp) -> 'Tensor':
    def softmax_backward(gradient):
        grad_array = gradient.data

        indices = np.where(grad_array == grad_array.max())

        arr = -grad_array * grad_array
        arr[indices] = grad_array[indices] * (1 - grad_array[indices])

        _set_grad(inp, arr)

    e = np.exp(inp.data - inp.data.max(axis=1, keepdims=True))
    z = e / np.sum(e, axis=1, keepdims=True)
    return _create_tensor(inp, data=z, func=softmax_backward)


def tanh(inp) -> 'Tensor':
    def tanh_backward(gradient):
        _set_grad(inp, gradient.data * (1 - np.square(output_array)))

    output_array = np.tanh(inp.data)
    return _create_tensor(inp, data=output_array, func=tanh_backward)


def dense(inp, weight, bias) -> 'Tensor':
    def dense_backward(gradient):
        _set_grad(inp, np.dot(gradient.data, weight.data))
        _set_grad(weight, np.dot(gradient.data.T, inp.data))
        _set_grad(bias, np.sum(gradient.data, axis=0, keepdims=True))

    return _create_tensor(inp, weight, bias, data=np.dot(inp.data, weight.data.T) + bias.data, func=dense_backward)


def conv(inp, weight, bias, stride, padding) -> 'Tensor':
    def conv_backward(gradient):
        _padded_input_array = np.pad(array=inp.data, pad_width=((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
        _weight_array = weight.data
        _grad_array = gradient.data

        _, _, _input_height, input_width = inp.shape
        _, _, _kernel_height, _kernel_width = _weight_array.shape
        _, _, _output_height, _output_width = _grad_array.shape
        _output_array = np.zeros_like(_padded_input_array)

        _weight_grad = np.zeros_like(_weight_array)
        _bias_grad = _grad_array.sum(axis=(0, 2, 3))

        for _row in range(_output_height):
            for _column in range(_output_width):
                _output_array[:, :, _row * stride:_row * stride + _kernel_height, _column * stride:_column * stride + _kernel_width] += np.sum(
                    _weight_array[np.newaxis, :, :, :, :] *
                    _grad_array[:, :, np.newaxis, _row:_row + 1, _column:_column + 1],
                    axis=1
                )
                _weight_grad += np.sum(
                    _padded_input_array[:, np.newaxis, :, _row * stride:_row * stride + _kernel_height, _column * stride:_column * stride + _kernel_width] *
                    _grad_array[:, :, np.newaxis, _row:_row + 1, _column:_column + 1],
                    axis=0
                )

        _set_grad(inp, _weight_grad)
        _set_grad(weight, _bias_grad)
        _set_grad(bias, _output_array[:, :, padding[0]:padding[0] + _input_height, padding[1]:padding[1] + input_width])

    padded_input_array = np.pad(array=inp.data, pad_width=((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
    weight_array = weight.data
    bias_array = bias.data

    output_shape = _calculate_output_dims(
        input_shape=inp.shape,
        kernel_shape=weight.shape,
        padding=padding,
        stride=stride
    )

    output_array = np.zeros(output_shape)

    _, _, kernel_height, kernel_width = weight.shape
    _, _, output_height, output_width = output_shape

    for row in range(output_height):
        for column in range(output_width):
            output_array[:, :, row, column] = np.sum(
                padded_input_array[:, np.newaxis, :, row * stride:row * stride + kernel_height, column * stride:column * stride + kernel_width] *
                weight_array[np.newaxis, :, :, :],
                axis=(2, 3, 4)
            )

    return _create_tensor(inp, weight, bias, data=output_array + bias_array[:, np.newaxis, np.newaxis], func=conv_backward)


def dropout(inp, keep_prob) -> 'Tensor':
    def apply_mask(array) -> np.array:
        array *= mask
        array /= keep_prob
        return array

    def dropout_backward(gradient):
        _set_grad(inp, apply_mask(gradient))

    mask = (np.random.rand(*inp.shape) < keep_prob)
    out = apply_mask(inp.data)
    return _create_tensor(inp, data=out, func=dropout_backward)


def batch_norm(inp, weight, bias, running_mean, running_var, momentum, eps, training) -> 'Tensor':
    def batch_norm_backward(gradient):
        if training:
            batch_size = inp.data.shape[0]
            weight_by_grad = weight.data * gradient.data
            dxc = weight_by_grad / input_standard_deviation
            dstd = -np.sum((weight_by_grad * input_mean_difference) / (input_standard_deviation * input_standard_deviation), axis=0)
            dvar = 0.5 * dstd / input_standard_deviation
            dxc += (2.0 / batch_size) * input_mean_difference * dvar
            dmu = np.sum(dxc, axis=0)

            _set_grad(inp, dxc - dmu / batch_size)
            _set_grad(weight, np.sum(input_mean_over_input_standard_deviation * gradient.data, axis=0))
            _set_grad(bias, gradient.data.sum(axis=0))
        else:
            weight_by_grad = weight.data * gradient.data

            _set_grad(inp, weight_by_grad / input_standard_deviation)
            _set_grad(weight, np.sum(input_mean_over_input_standard_deviation * gradient.data, axis=0))
            _set_grad(bias, gradient.data.sum(axis=0))

    input_array = inp.data
    running_mean_array = running_mean.data
    running_var_array = running_var.data
    gamma_array = weight.data
    beta_array = bias.data

    if len(input_array.shape) == 2:
        input_mean = np.mean(input_array, axis=0)

        input_mean_difference = input_array - input_mean
        input_variance = np.mean(input_mean_difference ** 2, axis=0)
        input_standard_deviation = np.sqrt(input_variance)
        input_standard_deviation[input_standard_deviation == 0] = input_standard_deviation[input_standard_deviation == 0] + eps
        input_mean_over_input_standard_deviation = input_mean_difference / input_standard_deviation

        if training:
            input_variance[input_variance == 0] = input_variance[input_variance == 0] + eps
            x_hat = (input_array - input_mean) / np.sqrt(input_variance)
        else:
            running_var_array[running_var_array == 0] = running_var_array[running_var_array == 0] + eps
            x_hat = (input_array - running_mean_array) / np.sqrt(running_var_array)
        out = gamma_array * x_hat + beta_array
    elif len(input_array.shape) == 4:
        _, channel, _, _ = input_array.shape
        input_mean = np.mean(input_array, axis=(0, 2, 3))

        input_mean_difference = input_array - input_mean.reshape((1, channel, 1, 1))
        input_variance = np.mean(input_mean_difference ** 2, axis=(0, 2, 3))
        input_standard_deviation = np.sqrt(input_variance.reshape((1, channel, 1, 1)))
        input_standard_deviation[input_standard_deviation == 0] = input_standard_deviation[input_standard_deviation == 0] + eps
        input_mean_over_input_standard_deviation = input_mean_difference / input_standard_deviation

        if training:
            input_variance[input_variance == 0] = input_variance[input_variance == 0] + eps
            x_hat = (input_array - input_mean.reshape((1, channel, 1, 1))) / np.sqrt(input_variance.reshape((1, channel, 1, 1)))
        else:
            running_var_array[running_var_array == 0] = running_var_array[running_var_array == 0] + eps
            x_hat = (input_array - running_mean_array.reshape((1, channel, 1, 1))) / np.sqrt(running_var_array.reshape((1, channel, 1, 1)))
        out = gamma_array.reshape((1, channel, 1, 1)) * x_hat + beta_array.reshape((1, channel, 1, 1))
    else:
        raise ValueError

    if training:
        running_mean.data = running_mean_array * (1.0 - momentum) + input_mean * momentum
        running_var.data = running_var_array * (1.0 - momentum) + input_variance * momentum

    return _create_tensor(inp, weight, bias, data=out, func=batch_norm_backward)


def max_pool(inp, kernel_size, stride, padding) -> 'Tensor':
    def save_mask(x, cords):
        mask = np.zeros_like(x)
        n, c, h, w = x.shape
        x = x.reshape(n, h * w, c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape((n, h * w, c))[n_idx, idx, c_idx] = 1
        cache[cords] = mask

    def max_pool_backward(gradient):
        grad_array = gradient.data

        _, _, _output_height, _output_width = grad_array.shape
        _kernel_height, _kernel_width = kernel_size

        _padded_input_array = np.pad(array=inp.data, pad_width=((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
        _output_array = np.zeros_like(_padded_input_array)

        for _row in range(_output_height):
            for _column in range(_output_width):
                increment = grad_array[:, :, _row:_row + 1, _column:_column + 1] * cache[(_row, _column)]
                _output_array[:, :, _row * stride:_row * stride + _kernel_height, _column * stride:_column * stride + _kernel_width] += increment

        _set_grad(inp, _output_array[:, :, padding[0]:padding[0] + _output_height - 1, padding[1]:padding[1] + _output_width - 1])

    cache = {}

    padded_input_array = np.pad(array=inp.data, pad_width=((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')

    _, _, output_height, output_width = _calculate_output_dims(inp.shape, (0, 0, kernel_size[0], kernel_size[1]), padding, stride)
    kernel_height, kernel_width = kernel_size
    batch_size, channels, _, _ = padded_input_array.shape

    output_array = np.zeros((batch_size, channels, output_height, output_width))

    for row in range(output_height):
        for column in range(output_width):
            padded_input_slice = padded_input_array[:, :, row * stride:row * stride + kernel_height, column * stride:column * stride + kernel_width]
            save_mask(x=padded_input_slice, cords=(row, column))
            output_array[:, :, row, column] = np.max(padded_input_slice, axis=(2, 3))

    return _create_tensor(inp, data=output_array, func=max_pool_backward)


def avg_pool(inp, kernel_size, stride, padding) -> 'Tensor':
    # pool should have mode
    def avg_pool_backward(gradient):
        grad_array = gradient.data

        _, _, _output_height, _output_width = grad_array.shape
        _kernel_height, _kernel_width = kernel_size

        _padded_input_array = np.pad(array=inp.data, pad_width=((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
        _output_array = np.zeros_like(_padded_input_array)

        for _row in range(_output_height):
            for _column in range(_output_width):
                increment = grad_array[:, :, _row:_row + 1, _column:_column + 1] / _kernel_height / _kernel_width
                _output_array[:, :, _row * stride:_row * stride + _kernel_height, _column * stride:_column * stride + _kernel_width] += increment

        _set_grad(inp, _output_array[:, :, padding[0]:padding[0] + _output_height - 1, padding[1]:padding[1] + _output_width - 1])

    padded_input_array = np.pad(array=inp.data, pad_width=((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')

    _, _, output_height, output_width = _calculate_output_dims(inp.shape, (0, 0, kernel_size[0], kernel_size[1]), padding, stride)
    kernel_height, kernel_width = kernel_size
    batch_size, channels, _, _ = padded_input_array.shape

    output_array = np.zeros((batch_size, channels, output_height, output_width))

    for row in range(output_height):
        for column in range(output_width):
            padded_input_slice = padded_input_array[:, :, row * stride:row * stride + kernel_height, column * stride:column * stride + kernel_width]
            output_array[:, :, row, column] = np.mean(padded_input_slice, axis=(2, 3))

    return _create_tensor(inp, data=output_array, func=avg_pool_backward)


def rnn_relu(inp, hx, all_weights, bias, num_layers, drop, training):
    # inp.shape = b, t, f
    # hx.shape = n, b, h
    def rnn_relu_backward(gradient):
        _inp_grads = np.zeros_like(inp.data)

        for _layer in range(num_layers):
            _w_ih, _w_hh, _b_ih, _b_hh = all_weights[_layer]

            _w_ih_grads = np.zeros_like(_w_ih.data)
            _w_hh_grads = np.zeros_like(_w_hh.data)
            _b_ih_grads = np.zeros_like(_b_ih.data)
            _b_hh_grads = np.zeros_like(_b_hh.data)

            for _time in range(inp.size(1) - 1, -1, -1):
                _prev_h = intermediate_values[_time][_layer]['prev_h']
                _current_h = intermediate_values[_time][_layer]['current_h']
                _inp = intermediate_values[_time][_layer]['input']
                _relu_input = intermediate_values[_time][_layer]['relu_input']

                _w_ih_power = np.power(_w_ih.data, inp.size(1) - _time - 1)
                _w_hh_power = np.power(_w_hh.data, inp.size(1) - _time - 1)

                _out = np.zeros_like(_relu_input.data)
                _out[_relu_input.data <= 0] = 0
                _out[_relu_input.data > 0] = 1

                # gradient._data = gradient.data * out

                _w_ih_grads += np.dot((_out * gradient.data[:, _time, :]).T, _inp.data) * _w_ih_power
                _w_hh_grads += np.dot((_out * gradient.data[:, _time, :]).T, _prev_h.data) * _w_hh_power
                _b_ih_grads += np.sum((_out * gradient.data[:, _time, :]), axis=0)
                _b_hh_grads += np.sum((_out * gradient.data[:, _time, :]), axis=0)

                print(
                    out.shape,
                    gradient.shape,
                    w_ih.shape,
                    (_out * gradient.data[:, _time, :]).shape,
                    np.dot((_out * gradient.data[:, _time, :]), _w_ih.data).shape,
                    _inp_grads[:, _time, :].shape
                )
                _inp_grads[:, _time, :] += np.dot((_out * gradient.data[:, _time, :]), _w_ih.data)

            _set_grad(_w_ih, data=_w_ih_grads)
            _set_grad(_w_hh, data=_w_hh_grads)
            _set_grad(_b_ih, data=_b_ih_grads)
            _set_grad(_b_hh, data=_b_hh_grads)
        _set_grad(inp, data=_inp_grads)

    out_tensor = zeros((inp.size(0), inp.size(1), hx.size(2)))

    intermediate_values = {}
    for time in range(inp.size(1)):
        if time not in intermediate_values:
            intermediate_values[time] = {}

        out = inp[:, time, :]
        for layer in range(num_layers):
            if layer not in intermediate_values[time]:
                intermediate_values[time][layer] = {}

            if bias:
                w_ih, w_hh, b_ih, b_hh = all_weights[layer]
            else:
                w_ih, w_hh = all_weights[layer]
                b_ih, b_hh = None, None

            h = hx[layer]

            intermediate_values[time][layer]['input'] = out
            out = dense(out, w_ih, b_ih) + dense(h, w_hh, b_hh)
            intermediate_values[time][layer]['relu_input'] = out
            out = relu(out)
            intermediate_values[time][layer]['prev_h'] = h
            intermediate_values[time][layer]['current_h'] = out
            hx[layer] = out

        out_tensor[:, time, :] = out

    from functools import reduce
    out_tensor = _create_tensor(inp, *reduce(lambda x, y: x + y, all_weights), data=out_tensor.data, func=rnn_relu_backward)
    return out_tensor, hx


def rnn_tanh(inp, hx, all_weights, bias, num_layers, drop, training):
    # inp.shape = b, t, f
    # hx.shape = n, b, h
    def rnn_tanh_backward(gradient):
        inp.backward()
        h.backward()
        w_ih.backward()
        w_hh.backward()
        if b_ih is not None:
            b_ih.backward()
        if b_hh is not None:
            b_hh.backward()

    out_tensor = zeros((inp.size(0), inp.size(1), hx.size(2)))
    for time in range(inp.size(1)):
        out = inp[:, time, :]
        for layer in range(num_layers):
            if bias:
                w_ih, w_hh, b_ih, b_hh = all_weights[layer]
            else:
                w_ih, w_hh = all_weights[layer]
                b_ih, b_hh = None, None

            h = hx[layer]

            out._data = tanh(dense(out, w_ih, b_ih) + dense(h, w_hh, b_hh)).data

            h._data = out.data

            hx[layer] = h.data  # need to stack, instead of assigning

        out_tensor[:, time, :] = out.data

    out_tensor = _create_tensor(inp, data=out_tensor.data, func=rnn_tanh_backward)
    return out_tensor, hx


def lstm(inp, hx, all_weights, bias, num_layers, drop, training):
    # inp.shape = b, t, f
    # hx.shape = n, b, h
    # cx.shape = n, b, h
    def lstm_backward(gradient):
        inp.backward()
        h.backward()
        w_ih.backward()
        w_hh.backward()
        if b_ih is not None:
            b_ih.backward()
        if b_hh is not None:
            b_hh.backward()

    hx, cx = hx
    out_tensor = zeros((inp.size(0), inp.size(1), hx.size(2)))
    for time in range(inp.size(1)):
        out = inp[:, time, :]
        for layer in range(num_layers):
            if bias:
                w_ih, w_hh, b_ih, b_hh = all_weights[layer]
            else:
                w_ih, w_hh = all_weights[layer]
                b_ih, b_hh = None, None

            h = hx[layer]
            c = cx[layer]

            gates = dense(out, w_ih, b_ih) + dense(h, w_hh, b_hh)

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = sigmoid(ingate)
            forgetgate = sigmoid(forgetgate)
            cellgate = tanh(cellgate)
            outgate = sigmoid(outgate)

            c._data = ((forgetgate * c) + (ingate * cellgate)).data
            out._data = (outgate * tanh(c)).data

            h._data = out.data

            cx[layer] = c.data
            hx[layer] = h.data  # need to stack, instead of assigning

        out_tensor[:, time, :] = out.data

    out_tensor = _create_tensor(inp, data=out_tensor.data, func=lstm_backward)
    return out_tensor, (hx, cx)


def gru(inp, hx, all_weights, bias, num_layers, drop, training):
    # inp.shape = b, t, f
    # hx.shape = n, b, h
    def gru_backward(gradient):
        inp.backward()
        h.backward()
        w_ih.backward()
        w_hh.backward()
        if b_ih is not None:
            b_ih.backward()
        if b_hh is not None:
            b_hh.backward()

    out_tensor = zeros((inp.size(0), inp.size(1), hx.size(2)))
    for time in range(inp.size(1)):
        out = inp[:, time, :]
        for layer in range(num_layers):
            if bias:
                w_ih, w_hh, b_ih, b_hh = all_weights[layer]
            else:
                w_ih, w_hh = all_weights[layer]
                b_ih, b_hh = None, None

            h = hx[layer]

            gi = dense(out, w_ih, b_ih)
            gh = dense(h, w_hh, b_hh)
            i_r, i_i, i_n = gi.chunk(3, 1)
            h_r, h_i, h_n = gh.chunk(3, 1)

            resetgate = sigmoid(i_r + h_r)
            inputgate = sigmoid(i_i + h_i)
            newgate = tanh(i_n + resetgate * h_n)
            out._data = (newgate + inputgate * (h - newgate)).data

            hx[layer] = out  # need to stack, instead of assigning

        out_tensor[:, time, :] = out

    out_tensor = _create_tensor(inp, data=out_tensor.data, func=gru_backward)
    return out_tensor, hx


def rnn_relu_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    def rnn_relu_cell_backward(gradient):
        inp.backward()
        h.backward()
        w_ih.backward()
        w_hh.backward()
        if b_ih is not None:
            b_ih.backward()
        if b_hh is not None:
            b_hh.backward()

    return relu(dense(inp, w_ih, b_ih) + dense(h, w_hh, b_hh))


def rnn_tanh_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    def rnn_tanh_cell_backward(gradient):
        inp.backward()
        h.backward()
        w_ih.backward()
        w_hh.backward()
        if b_ih is not None:
            b_ih.backward()
        if b_hh is not None:
            b_hh.backward()

    return tanh(dense(inp, w_ih, b_ih) + dense(h, w_hh, b_hh))


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
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad._data = grad.data + param.data * weight_decay

        # Decay the first and second moment running average coefficient
        exp_avg._data = exp_avg.data * beta1 + (1 - beta1) * grad.data
        exp_avg_sq._data = exp_avg_sq.data * beta2 + (1 - beta2) * (grad.data * grad.data)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sq._data = np.maximum(max_exp_avg_sq.data, exp_avg_sq.data)
            # Use the max. for normalizing running avg. of gradient
            denom = (np.sqrt(max_exp_avg_sq.data) / math.sqrt(bias_correction2)) + eps
        else:
            denom = (np.sqrt(exp_avg_sq.data) / math.sqrt(bias_correction2)) + eps

        step_size = lr / bias_correction1

        param._data = param.data - step_size * (exp_avg.data / denom)


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
