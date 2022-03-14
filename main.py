from array import array
from mocha import *


@compiled()
def add(a: list[int], b: int):
    b = 1
    a[2] = b
    if b > 1 and b < 2 and b > 5:
        a[3] = 3
    else:
        a[3] = 4
