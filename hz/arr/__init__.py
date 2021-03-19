from .functional import (
    add,
    div,
    mul,
    sub,
    power
)
from .ndarray import NDArray

NDArray.add = add
NDArray.sub = sub
NDArray.div = div
NDArray.mul = mul
NDArray.power = power
