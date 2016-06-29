#!/usr/bin/env python
import numpy
import kernel_tuner
from collections import OrderedDict

with open('pnpoly.cu', 'r') as f:
    kernel_string = f.read()

problem_size = (20000000, 1)
size = numpy.int32(numpy.prod(problem_size))
vertices = 600

points = numpy.random.randn(2*size).astype(numpy.float32)
bitmap = numpy.zeros(size).astype(numpy.int32)
args = [bitmap, points, size]

cmem_args= {'d_Vertices': numpy.random.randn(2*vertices).astype(numpy.float32) }

tune_params = OrderedDict()
tune_params["block_size_x"] = [2**i for i in range(1,10)][::-1]
tune_params["prefetch"] = [0, 1]
tune_params["use_bitmap"] = [0, 1, 2]
tune_params["coalesce_bitmap"] = [0, 1]
tune_params["tile_size"] = [1] + [2*i for i in range(1,17)]

#use restriction because coalesce_bitmap=1 is only effective when use_bitmap=1
restrict = ["coalesce_bitmap==0 or use_bitmap==1"]

grid_div_x = ["block_size_x", "tile_size"]

kernel_tuner.tune_kernel("cn_PnPoly", kernel_string,
    problem_size, args, tune_params,
    grid_div_x=grid_div_x, restrictions=restrict, cmem_args=cmem_args)

