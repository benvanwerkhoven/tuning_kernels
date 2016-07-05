#!/usr/bin/env python

import numpy
from kernel_tuner import tune_kernel, run_kernel

with open('maxloc.cu', 'r') as f:
    kernel_string = f.read()

tune_params = dict()
tune_params["block_size_x"] = [64]

size = 32 * 1024
num_blocks = numpy.int32(8)
problem_size = ("num_blocks", 1)

x = numpy.random.rand(size).astype(numpy.float32)

max_loc = numpy.zeros(num_blocks).astype(numpy.int32)
max_temp = numpy.zeros(num_blocks).astype(numpy.float32)
locations = numpy.zeros(size).astype(numpy.float32)
use_index = numpy.int32(1)
n = numpy.int32(size)

args = [max_loc, max_temp, locations, x, use_index, n]

params = dict()
params["block_size_x"] = 64
params["num_blocks"] = 8
params["use_shuffle"] = 1
params["vector"] = 4

#call the first kernel that computes the incomplete max locs
result = run_kernel("max_loc", kernel_string,
    problem_size, args, params,
    grid_div_x=[])

#then call the kernel again on the intermediate result with 1 thread block
args = [max_loc, max_temp, result[0], result[1], numpy.int32(0), num_blocks]

params["num_blocks"] = 1

result_final = run_kernel("max_loc", kernel_string,
    (1, 1), args, params,
    grid_div_x=[])

print "expected", numpy.argmax(x), x.max()
print "intermediate answer", result[0], result[1]
print "kernel answer", result_final[0][0], result_final[1][0]
