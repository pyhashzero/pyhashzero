from .functional import (
    add,
    div,
    eq,
    ge,
    gt,
    le,
    lt,
    mul,
    ne,
    negative,
    power,
    sub
)
from .ndarray import NDArray

NDArray.add = add
NDArray.sub = sub
NDArray.div = div
NDArray.mul = mul
NDArray.power = power
NDArray.negative = negative
NDArray.lt = lt
NDArray.le = le
NDArray.gt = gt
NDArray.ge = ge
NDArray.eq = eq
NDArray.ne = ne
