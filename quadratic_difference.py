#!/usr/bin/env python

import numpy as np
from kernel_tuner import tune_kernel, run_kernel

with open('quadratic_difference.cu', 'r') as f:
    kernel_string = f.read()

N = np.int32(4.5e6)
#N = np.int32(30000)
print (N)

N_light_crossing     = 1500
sliding_window_width = np.int32(N_light_crossing)

problem_size = (int(N), int(sliding_window_width))
print (problem_size)

#x = np.random.random(N).astype(np.float32)
#y = np.random.random(N).astype(np.float32)
#z = np.random.random(N).astype(np.float32)
#ct = np.random.random(N).astype(np.float32)

#input data resulting in more realistic hit density
x = np.random.normal(0.2, 0.1, N).astype(np.float32)
y = np.random.normal(0.2, 0.1, N).astype(np.float32)
z = np.random.normal(0.2, 0.1, N).astype(np.float32)
ct = 1000*np.random.normal(0.5, 0.06, N).astype(np.float32)

correlations = np.zeros((sliding_window_width, N), 'uint8')
sums = np.zeros(N).astype(np.int32)

args_old = [correlations, N, sliding_window_width, x, y, z, ct]
args = [correlations, sums, N, sliding_window_width, x, y, z, ct]

tune_params = dict()
tune_params["block_size_x"] = [2**i for i in range(7)]
tune_params["block_size_y"] = [2**i for i in range(7)]

grid_div_x = ["block_size_x"]
grid_div_y = ["block_size_y"]

restrict = ["block_size_x*block_size_y >= 32"]

#run the kernel once for with parameters known to produce correct output
#the result list can be used to verify the output of the quadratic_difference_linear kernel
params = { "block_size_x": 16, "block_size_y": 16 }
result = run_kernel("quadratic_difference", kernel_string, problem_size, args_old, params, grid_div_x=grid_div_x, grid_div_y=grid_div_y)



#uncomment the following to tune the old kernel
#tune_kernel("quadratic_difference", kernel_string, problem_size, args, tune_params,
#    grid_div_x=grid_div_x, grid_div_y=grid_div_y, restrictions=restrict)



#now tune the quadratic_difference_linear kernel
kernel_name = "quadratic_difference_linear"


problem_size = (int(N), 1)

tune_params = dict()
tune_params["block_size_x"] = [32*i for i in range(1,33)] #multiples of 32
tune_params["f_unroll"] = [i for i in range(1,20) if 1500/float(i) == 1500//i] #divisors of 1500
tune_params["tile_size_x"] = [2**i for i in range(5)] #powers of 2

tune_params["write_sums"] = [1]
#tune_params["block_size_x"] = [2**i for i in range(7,11)]
#tune_params["tile_size_x"] = [1, 2, 4, 8, 16]
#tune_params["shmem"] = [0, 1]
#tune_params["read_only"] = [0]
#tune_params["use_if"] = [1]


if "tile_size_x" in tune_params:
    grid_div_x = ["block_size_x", "tile_size_x"]
else:
    grid_div_x = ["block_size_x"]

if 1 in tune_params["write_sums"]:
    result = [result[0], np.sum(result[0], axis=0).astype(np.int32), None, None, None, None, None, None]
    print("shape of sum array", result[1].shape)
    print("shape of correlations", result[0].shape)
    print("sum of correlations, sum of row sums", np.sum(result[0]), np.sum(result[1]) )

else:
    result = [result[0], None, None, None, None, None, None, None]


#set False here to True to visually compare the output of both kernels
if False:
    params = dict()
    params["block_size_x"] = 512
    params["use_if"] = 1
    params["tile_size_x"] = 2
    params["write_sums"] = 1
    result2 = run_kernel(kernel_name, kernel_string, problem_size, args, params,
        grid_div_x=grid_div_x)

    print ("sum hits ", np.sum(result[1]), np.sum(result2[1]) )

if False:
    from matplotlib import pyplot
    #corr1 = result[0].reshape(1500, N)
    #corr2 = result2[0].reshape(1500, N)
    corr1 = result[1].reshape(N/150, 150)
    corr2 = result2[1].reshape(N/150, 150)
    f, (ax1, ax2, ax3) = pyplot.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    ax1.imshow(corr1, cmap=pyplot.cm.bone)
    ax2.imshow(corr2, cmap=pyplot.cm.bone)
    ax3.imshow(corr2-corr1, cmap=pyplot.cm.jet)
    pyplot.show()





#because of a bug in PyCuda we can't automatically verify the result of arrays larger than 2**32
if correlations.nbytes > 2**32:
    tune_kernel(kernel_name, kernel_string, problem_size, args, tune_params,
        grid_div_x=grid_div_x, verbose=True)
else:
    tune_kernel(kernel_name, kernel_string, problem_size, args, tune_params,
        grid_div_x=grid_div_x, answer=result, verbose=True)

