#!/usr/bin/env python
import numpy
import kernel_tuner
from collections import OrderedDict

with open('pnpoly.cu', 'r') as f:
    kernel_string = f.read()

problem_size = (20000, 1)
size = numpy.int32(numpy.prod(problem_size))
vertices = 600

points = numpy.random.randn(2*size).astype(numpy.float32)
bitmap = numpy.zeros(size).astype(numpy.int32)
args = [bitmap, points, size]

#the vertices are not yet sorted so they probably do not form a proper polygon yet
#need to verify the output of the gpu kernel, with a circle I can do a simple distance to 0,0 check for all points

#should use pyplot.plot to plot the vertices and see if they are in the right order to form a polygon
#don't know yet if it should be clockwise or counter-clockwise

vertex_seeds = numpy.sort(numpy.random.rand(vertices)*2.0*numpy.pi)

points_x = points[::2]
points_y = points[1::2]

print "points_x min max", points_x.min(), points_x.max()
print "points_y min max", points_y.min(), points_y.max()

vertex_x = numpy.cos(vertex_seeds)
vertex_y = numpy.sin(vertex_seeds)
vertex_xy = numpy.array( zip(vertex_x, vertex_y) ).astype(numpy.float32)

print "vertex_x min max", vertex_x.min(), vertex_x.max()
print "vertex_y min max", vertex_y.min(), vertex_y.max()



#from matplotlib import pyplot
#pyplot.scatter(points_x, points_y)
#pyplot.plot(vertex_x, vertex_y)
#pyplot.show()

#raw_input()


cmem_args= {'d_Vertices': vertex_xy }


tune_params = OrderedDict()
tune_params["block_size_x"] = [2**i for i in range(1,10)][::-1]
tune_params["prefetch"] = [0, 1]
tune_params["use_bitmap"] = [0, 1, 2]
tune_params["coalesce_bitmap"] = [0, 1]
tune_params["tile_size"] = [1] + [2*i for i in range(1,17)]

#use restriction because coalesce_bitmap=1 is only effective when use_bitmap=1
restrict = ["coalesce_bitmap==0 or use_bitmap==1"]

grid_div_x = ["block_size_x", "tile_size"]

params = dict()
params["block_size_x"] = 512
params["prefetch"] = 0
params["use_bitmap"] = 0
params["coalesce_bitmap"] = 0
params["tile_size"] = 1

result = kernel_tuner.run_kernel("cn_PnPoly_naive", kernel_string,
    problem_size, args, params,
    grid_div_x=grid_div_x, cmem_args=cmem_args)

print "sum=" + str(numpy.sum(result[0]))


#compute reference answer

reference = [numpy.sqrt(x*x + y*y) < 1.0 for x,y in zip(points_x, points_y)]
print "answer=" + str(numpy.sum(reference))



#kernel_tuner.tune_kernel("cn_PnPoly", kernel_string,
#    problem_size, args, tune_params,
#    grid_div_x=grid_div_x, restrictions=restrict, cmem_args=cmem_args)



