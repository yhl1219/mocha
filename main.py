from mocha import *


a = Field(float, (1, 2))
b = Field(float, (3, 4))

def st(coord):
    a[coord.x, coord.y] = b[coord.x, coord.y]


stencil(st)
