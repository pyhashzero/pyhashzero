from .functional import (
    add,
    copy,
    div,
    eq,
    fill,
    ge,
    gt,
    le,
    lt,
    mul,
    ne,
    negative,
    power,
    reshape,
    sub
)
from .array import Array

Array.add = add
Array.sub = sub
Array.div = div
Array.mul = mul
Array.power = power
Array.negative = negative
Array.lt = lt
Array.le = le
Array.gt = gt
Array.ge = ge
Array.eq = eq
Array.ne = ne
Array.reshape = reshape
Array.fill = fill
Array.copy = copy
