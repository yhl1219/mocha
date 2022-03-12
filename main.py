from mocha import *


@compiled()
def add(a: matrix, b: int):
    a[1, 2] = b