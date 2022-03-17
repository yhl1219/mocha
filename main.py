from mocha import *


a = Field(float, (1, 2))


def st(coord):
    a[3, 4] = 3


stencil(st)
