from H0.nn.core import Tensor
from H0.nn.functional import (
    abs,
    adam,
    add,
    arange,
    avg_pool,
    batch_norm,
    ceil,
    chunk,
    clip,
    clone,
    concat,
    conv,
    cpu,
    dense,
    detach,
    div,
    double,
    dropout,
    empty,
    empty_like,
    expand_dim,
    eye,
    eye_like,
    fill,
    floor,
    from_array,
    full,
    full_like,
    gpu,
    half,
    index_select,
    is_tensor,
    linspace,
    max_pool,
    mean,
    mul,
    negative,
    normal,
    normal_like,
    one,
    ones,
    ones_like,
    pow,
    rand,
    rand_like,
    randint,
    randint_like,
    randn,
    randn_like,
    relu,
    round,
    sigmoid,
    single,
    softmax,
    squeeze,
    stack,
    std,
    sub,
    sum,
    tanh,
    to_array,
    transpose,
    uniform,
    uniform_like,
    var,
    view,
    zero,
    zeros,
    zeros_like
)

Tensor.chunk = chunk
Tensor.view = view
Tensor.index_select = index_select

Tensor.zero = zero
Tensor.one = one
Tensor.fill = fill
Tensor.squeeze = squeeze
Tensor.expand_dim = expand_dim
Tensor.transpose = transpose
Tensor.abs = abs
Tensor.round = round
Tensor.floor = floor
Tensor.ceil = ceil
Tensor.clip = clip
Tensor.negative = negative
Tensor.sum = sum
Tensor.mean = mean
Tensor.std = std
Tensor.var = var
Tensor.var = var
Tensor.add = add
Tensor.sub = sub
Tensor.mul = mul
Tensor.div = div
Tensor.pow = pow
Tensor.clone = clone
Tensor.detach = detach

Tensor.from_array = from_array
Tensor.to_array = to_array

Tensor.half = half
Tensor.single = single
Tensor.double = double
Tensor.cpu = cpu
Tensor.gpu = gpu
