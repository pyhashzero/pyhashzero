from hz.ai.nn.core import Tensor
from hz.ai.nn.functional import *

Tensor.chunk = chunk
Tensor.view = view
Tensor.index_select = index_select

Tensor.zero = zero
Tensor.one = one
Tensor.fill = fill
Tensor.squeeze = squeeze
Tensor.expand_dim = expand_dim
Tensor.transpose = transpose
Tensor.absolute = absolute
Tensor.around = around
Tensor.floor = floor
Tensor.ceil = ceil
Tensor.clip = clip
Tensor.negative = negative
Tensor.summation = summation
Tensor.mean = mean
Tensor.std = std
Tensor.var = var
Tensor.var = var
Tensor.add = add
Tensor.sub = sub
Tensor.mul = mul
Tensor.div = div
Tensor.power = power
Tensor.clone = clone
Tensor.detach = detach

Tensor.from_array = from_array
Tensor.to_array = to_array

Tensor.half = half
Tensor.single = single
Tensor.double = double
Tensor.cpu = cpu
Tensor.gpu = gpu
