#include <stdio.h>

#ifndef block_size_x
#define block_size_x 256
#endif

__device__ __forceinline__ float prefix_sum_warp(float v, int end) {
    float x = v;

    asm("{                  \n\t"
        "  .reg .f32  t;    \n\t"
        "  .reg .pred p;    \n\t");

    #pragma unroll
    for (unsigned int d=1; d<end; d*=2) {
        asm("shfl.up.b32 t|p, %0, %1, 0x0;  \n\t"
            "@p add.f32 %0, %0, t;          \n\t" : "+f"(x) : "r"(d));
    }

    asm("}");

    return x;
}

__global__ void prefix_sum_block(float *prefix_sums, float *block_carry, float *input, int n) {
    int tx = threadIdx.x;
    int x = blockIdx.x * block_size_x + tx;
    float v = 0.0f;
    if (x < n) {
        v = input[blockIdx.x * block_size_x + tx];
    }

    v = prefix_sum_warp(v, 32);

    #if block_size_x > 32
    int laneid = tx & (32-1);
    int warpid = tx>>5; // /32

    __shared__ float warp_carry[block_size_x/32];
    if (laneid == 31) {
        warp_carry[warpid] = v;
    }
    __syncthreads();

    if (tx < block_size_x/32) {
        float temp = warp_carry[tx];
        prefix_sum_warp(temp, block_size_x/32);
        warp_carry[tx] = temp;
    }
    __syncthreads();

    if (warpid>0) {
        v += warp_carry[warpid-1];
    }
    #endif

    //printf("thread %d, value %f, n=%d\n",tx,v,n);
    if (x < n) {
        prefix_sums[x] = v;
    }

    if (tx == block_size_x-1) {
        block_carry[blockIdx.x] = v;
    }
}


__global__ void propagate_block_carry(float *prefix_sums, float *block_carry, int n) {
    int x = blockIdx.x * block_size_x + threadIdx.x;
    if (blockIdx.x > 0 && x < n) {
        prefix_sums[x] = prefix_sums[x] + block_carry[blockIdx.x-1];
    }
}
