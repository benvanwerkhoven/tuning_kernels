#!/usr/bin/env python

import numpy as np
from kernel_tuner import tune_kernel, run_kernel

with open('quadratic_difference.cu', 'r') as f:
    kernel_string = f.read()

N = np.int32(4.5e6)
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

correlations = np.zeros((N, sliding_window_width), 'uint8')

args = [correlations, N, sliding_window_width, x, y, z, ct]

tune_params = dict()
tune_params["block_size_x"] = [2**i for i in range(7)]
tune_params["block_size_y"] = [2**i for i in range(7)]

grid_div_x = ["block_size_x"]
grid_div_y = ["block_size_y"]

restrict = ["block_size_x*block_size_y >= 32"]

#run the kernel once for with parameters known to produce correct output
#the result list can be used to verify the output of the quadratic_difference_linear kernel
params = { "block_size_x": 16, "block_size_y": 16 }
result = run_kernel("quadratic_difference", kernel_string, problem_size, args, params, grid_div_x=grid_div_x, grid_div_y=grid_div_y)
result = [result[0], None, None, None, None, None, None]

#uncomment the following to tune the old kernel
#tune_kernel("quadratic_difference", kernel_string, problem_size, args, tune_params,
#    grid_div_x=grid_div_x, grid_div_y=grid_div_y, restrictions=restrict)



#now tune my the quadratic_difference_linear kernel

problem_size = (int(N), 1)

tune_params = dict()
tune_params["block_size_x"] = [32*i for i in range(1,33)]
tune_params["use_if"] = [1]

grid_div_x = ["block_size_x"]

params = dict()
params["block_size_x"] = 32
params["use_if"] = 1

result2 = run_kernel("quadratic_difference_linear", kernel_string, problem_size, args, params,
    grid_div_x=grid_div_x)


#print ("hits ", np.sum(result[0]), np.sum(result2[0]), "density= ", np.sum(result[0])/float(correlations.size) )

#set False here to True to visually compare the output of both kernels
if False:
    from matplotlib import pyplot
    corr1 = result[0].reshape(1500, N)
    corr2 = result2[0].reshape(1500, N)
    f, (ax1, ax2) = pyplot.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    ax1.imshow(corr1, cmap=pyplot.cm.bone)
    ax2.imshow(corr2, cmap=pyplot.cm.bone)
    pyplot.show()




#because of a bug in PyCuda we can't automatically verify the result of arrays larger than 2**32
if correlations.nbytes > 2**32:
    tune_kernel("quadratic_difference_linear", kernel_string, problem_size, args, tune_params,
        grid_div_x=grid_div_x, verbose=True)
else:
    tune_kernel("quadratic_difference_linear", kernel_string, problem_size, args, tune_params,
        grid_div_x=grid_div_x, answer=result, verbose=True)

