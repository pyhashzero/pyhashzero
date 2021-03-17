from hz.arr.functional import (
    abs,
    add,
    arange,
    ceil,
    chunk,
    clip,
    clone,
    concat,
    cpu,
    div,
    double,
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
    is_ndarray,
    linspace,
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
    round,
    single,
    squeeze,
    stack,
    std,
    sub,
    sum,
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
from hz.arr.ndarray import NDArray

NDArray.chunk = chunk
NDArray.view = view
NDArray.index_select = index_select

NDArray.zero = zero
NDArray.one = one
NDArray.fill = fill
NDArray.squeeze = squeeze
NDArray.expand_dim = expand_dim
NDArray.transpose = transpose
NDArray.abs = abs
NDArray.round = round
NDArray.floor = floor
NDArray.ceil = ceil
NDArray.clip = clip
NDArray.negative = negative
NDArray.sum = sum
NDArray.mean = mean
NDArray.std = std
NDArray.var = var
NDArray.var = var
NDArray.add = add
NDArray.sub = sub
NDArray.mul = mul
NDArray.div = div
NDArray.pow = pow
NDArray.clone = clone

NDArray.from_array = from_array
NDArray.to_array = to_array

NDArray.half = half
NDArray.single = single
NDArray.double = double
NDArray.cpu = cpu
NDArray.gpu = gpu
