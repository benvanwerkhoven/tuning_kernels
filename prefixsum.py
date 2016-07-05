#!/usr/bin/env python

import numpy
from kernel_tuner import tune_kernel, run_kernel

with open('prefixsum.cu', 'r') as f:
    kernel_string = f.read()

tune_params = dict()
tune_params["block_size_x"] = [64]

size = 1024
problem_size = (size, 1)
max_blocks = int(numpy.ceil(size/float(max(tune_params["block_size_x"]))))

#x = numpy.random.rand(size).astype(numpy.float32)
x = numpy.ones(size).astype(numpy.float32)
print x

prefix_sums = numpy.zeros(size).astype(numpy.float32)
block_carry = numpy.zeros(max_blocks).astype(numpy.float32)
n = numpy.int32(size)

args = [prefix_sums, block_carry, x, n]

params = dict()
params["block_size_x"] = 64

#call the first kernel that computes the incomplete prefix sums
#and outputs the block carry values
result = run_kernel("prefix_sum_block", kernel_string,
    problem_size, args, params,
    grid_div_x=["block_size_x"])

prefix_sums = result[0]
print result[0]
print result[1]

block_filler = numpy.zeros(max_blocks).astype(numpy.float32)
block_out = numpy.zeros(max_blocks).astype(numpy.float32)

args = [block_out, block_filler, result[1], numpy.int32(max_blocks)]

#call the kernel again, but this time on the block carry values
#one thread block should be sufficient
if max_blocks > params["block_size_x"]:
    print("warning: block size too small")

result = run_kernel("prefix_sum_block", kernel_string,
    (1, 1), args, params,
    grid_div_x=[])

block_carry = result[0]
print result[0]
print result[1]

args = [prefix_sums, block_carry, n]

#call a simple kernel to propagate the block carry values to all
#elements
result = run_kernel("propagate_block_carry", kernel_string,
    problem_size, args, params,
    grid_div_x=["block_size_x"])

print result[0]
