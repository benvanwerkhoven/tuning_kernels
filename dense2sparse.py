#!/usr/bin/env python

import numpy as np
from kernel_tuner import tune_kernel, run_kernel

with open('dense2sparse.cu', 'r') as f:
    kernel_string = f.read()

#N = np.int32(4.5e6)
N = np.int32(30000)
print (N)

N_light_crossing     = 1500
sliding_window_width = np.int32(N_light_crossing)

problem_size = (int(N), 1)
print (problem_size)

#generate input with the expected density
correlations = np.random.randn((sliding_window_width * N))
correlations[correlations <= 2.87] = 0
correlations[correlations > 2.87] = 1
correlations = np.array(correlations.reshape(sliding_window_width, N), dtype=np.uint8)

print correlations.min(), correlations.max()

print "density", np.sum(correlations) / (N*sliding_window_width)

num_correlated_hits = np.sum(correlations)
sum_rows = np.sum(correlations, axis=0)


print "sum_rows max", sum_rows.max()

prefix_sum = np.cumsum(sum_rows).astype(np.int32)

print prefix_sum.shape
print prefix_sum

print "bliep", num_correlated_hits, np.sum(sum_rows)

row_idx = np.zeros(num_correlated_hits).astype(np.int32)
col_idx = np.zeros(num_correlated_hits).astype(np.int32)

args = [row_idx, col_idx, prefix_sum, correlations, N]

params = dict()
params["block_size_x"] = 256

result = run_kernel("dense2sparse_kernel", kernel_string, problem_size,
    args, params, grid_div_x=["block_size_x"])

row_idx = result[0]
col_idx = result[1]

correlations_restored = np.zeros((sliding_window_width, N), 'uint8')

correlations_restored[col_idx, row_idx] = 1
#for i in range(num_correlated_hits):
#    correlations_restored[col_idx[i], row_idx[i]] = 1


print "restored_hits", np.sum(correlations_restored)

if False:
    from matplotlib import pyplot
    corr1 = correlations
    corr2 = correlations_restored
    #corr1 = result[0].reshape(1500, N)
    #corr2 = result2[0].reshape(1500, N)
    #corr1 = result[1].reshape(N/150, 150)
    #corr2 = result2[1].reshape(N/150, 150)
    f, (ax1, ax2, ax3) = pyplot.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    ax1.imshow(corr1, cmap=pyplot.cm.bone)
    ax2.imshow(corr2, cmap=pyplot.cm.bone)
    ax3.imshow(corr2-corr1, cmap=pyplot.cm.jet)
    pyplot.show()



print all(correlations.ravel() - correlations_restored.ravel() == 0)
