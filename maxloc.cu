#include <stdio.h>

#ifndef vector
#define vector 1
#endif 

#if (vector==1)
#define floatvector float
#define intvector int
#elif (vector == 2)
#define floatvector float2
#define intvector int2
#elif (vector == 4)
#define floatvector float4
#define intvector int4
#endif

#if use_shuffle == 1
#define stop_loop 16
#elif use_shuffle == 0
#define stop_loop 0
#endif

#define MAX_LOC(old_v, new_v, old_idx, new_idx) \
    if (new_v > old_v) {                        \
        old_v = new_v;                          \
        old_idx = new_idx;                      \
    }                                           \

#define SET_LOC(loc, i) \

__global__ void max_loc(int *max_idx, float *max_val, intvector *location, floatvector *array, int use_index, int n) {
    int ti = threadIdx.x;
    int x = blockIdx.x * block_size_x + threadIdx.x;
    int step_size = num_blocks * block_size_x;
    float lmax = 0.0f;
    int lidx = 0;

    //cooperatively iterate over input array with all thread blocks
    for (int i=x; i<n/vector; i+=step_size) {
        floatvector v = array[i];
        intvector loc;
        if (use_index == 0) {
            loc = location[i];
        }
        else {
            #if vector == 1
            loc = i;
            #elif vector == 2
            loc.x = i*vector;
            loc.y = i*vector+1;
            #elif vector == 4
            loc.x = i*vector;
            loc.y = i*vector+1;
            loc.z = i*vector+2;
            loc.w = i*vector+3;
            #endif
        }
        #if vector == 1
        MAX_LOC(lmax, v, lidx, loc)
        #elif vector == 2
        MAX_LOC(lmax, v.x, lidx, loc.x)
        MAX_LOC(lmax, v.y, lidx, loc.y)
        #elif vector == 4
        MAX_LOC(lmax, v.x, lidx, loc.x)
        MAX_LOC(lmax, v.y, lidx, loc.y)
        MAX_LOC(lmax, v.z, lidx, loc.z)
        MAX_LOC(lmax, v.w, lidx, loc.w)
        #endif
    }
    
    //reduce sum to single value (or last 32 in case of use_shuffle)
    __shared__ float sh_max[block_size_x];
    __shared__ int sh_idx[block_size_x];
    sh_max[ti] = lmax;
    sh_idx[ti] = lidx;
    __syncthreads();
    #pragma unroll
    for (unsigned int s=block_size_x/2; s>stop_loop; s>>=1) {
        if (ti < s) {
            MAX_LOC(sh_max[ti], sh_max[ti + s], sh_idx[ti], sh_idx[ti+s])
        }
        __syncthreads();
    }

    //reduce last 32 values to single value using warp shuffle instructions
    #if use_shuffle == 1
    if (ti < 32) {
        lmax = sh_max[ti];
        lidx = sh_idx[ti];
        #pragma unroll
        for (unsigned int s=16; s>0; s>>=1) {
            float v = __shfl_down(lmax, s);
            int i = __shfl_down(lidx, s);
            MAX_LOC(lmax, v, lidx, i)
        }
    }
    #else
    if (ti == 0) {
        lmax = sh_max[0];
        lidx = sh_idx[0];
    }
    #endif

    //write back one value per thread block, run kernel again with one tread block
    if (ti == 0) {
        max_val[blockIdx.x] = lmax;
        max_idx[blockIdx.x] = lidx;
    }
}

