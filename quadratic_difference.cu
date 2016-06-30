#include <stdio.h>

#include <inttypes.h>

#ifndef tile_size_x
  #define tile_size_x 1
#endif

#ifndef block_size_y
  #define block_size_y 1
#endif

#define window_width 1500


__global__ void quadratic_difference(int8_t *correlations, int N, int sliding_window_width, float *x, float *y, float *z, float *ct)
{
    int i = blockIdx.x * block_size_x + threadIdx.x;
    int j = blockIdx.y * block_size_y + threadIdx.y;

    int l = i + j;
    if (i < N && j < sliding_window_width) { 

    //if you want to test the kernel with the old layout for
    //correlations add a tuning parameter with called "old" with [1]
    #if (old > 0)
      uint64_t pos = i * (uint64_t)sliding_window_width + j;
    #else
      uint64_t pos = j * (uint64_t)N + (uint64_t)i;
    #endif

    if (l >= N){
        return;
    }

    float diffct = ct[i] - ct[l];
    float diffx  = x[i] - x[l];
    float diffy  = y[i] - y[l];
    float diffz  = z[i] - z[l];

    if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) { 
      correlations[pos] = 1;
    }

    }
}


#ifndef tile_size_x
#define tile_size_x 1
#endif

#define USE_READ_ONLY_CACHE read_only
#if USE_READ_ONLY_CACHE == 1
#define LDG(x, y) __ldg(x+y)
#elif USE_READ_ONLY_CACHE == 0
#define LDG(x, y) x[y]
#endif


/*
 * This kernel uses the usual set of optimizations, including tiling, partial loop unrolling, read-only cache. 
 * Tuning parameters supported are 'read_only' [0,1], 'tile_size_x' divisor of 1500, and 'block_size_x' multiple of 32.
 *
 */
__global__ void quadratic_difference_linear(char *__restrict__ correlations, int N, int sliding_window_width,
        const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z, const float *__restrict__ ct) {

    int tx = threadIdx.x;
    int bx = blockIdx.x * block_size_x * tile_size_x;

    __shared__ float sh_ct[block_size_x * tile_size_x + window_width];
    __shared__ float sh_x[block_size_x * tile_size_x + window_width];
    __shared__ float sh_y[block_size_x * tile_size_x + window_width];
    __shared__ float sh_z[block_size_x * tile_size_x + window_width];

    if (bx+tx < N) {

        //the loading phase
        for (int k=tx; k < block_size_x*tile_size_x+window_width-1; k+=block_size_x) {
            if (bx+k < N) {
                sh_ct[k] = LDG(ct,bx+k);
                sh_x[k] = LDG(x,bx+k);
                sh_y[k] = LDG(y,bx+k);
                sh_z[k] = LDG(z,bx+k);
            }
        }
        __syncthreads();

        //start of the the computations phase
        int i = tx;
        float l_ct[tile_size_x];
        float l_x[tile_size_x];
        float l_y[tile_size_x];
        float l_z[tile_size_x];

        //keep the most often used values in registers
        for (int ti=0; ti<tile_size_x; ti++) {
            l_ct[ti] = sh_ct[i+ti*block_size_x];
            l_x[ti] = sh_x[i+ti*block_size_x];
            l_y[ti] = sh_y[i+ti*block_size_x];
            l_z[ti] = sh_z[i+ti*block_size_x];
        }

        //small optimization to eliminate bounds checks for most blocks
        if (bx+block_size_x*tile_size_x+window_width-1 < N) {

            //unfortunately there's no better way to do this right now
            //[1, 2, 3, 4, 5, 6, 10, 12, 15]
            #if f_unroll == 2
            #pragma unroll 2
            #elif f_unroll == 3
            #pragma unroll 3
            #elif f_unroll == 4
            #pragma unroll 4
            #elif f_unroll == 5
            #pragma unroll 5
            #elif f_unroll == 6
            #pragma unroll 6
            #elif f_unroll == 10
            #pragma unroll 10
            #elif f_unroll == 12
            #pragma unroll 12
            #elif f_unroll == 15
            #pragma unroll 15
            #endif            
            for (int j=0; j < window_width; j++) {

                #pragma unroll
                for (int ti=0; ti<tile_size_x; ti++) {

                        float diffct = l_ct[ti] - sh_ct[i+ti*block_size_x+j];
                        float diffx  = l_x[ti] - sh_x[i+ti*block_size_x+j];
                        float diffy  = l_y[ti] - sh_y[i+ti*block_size_x+j];
                        float diffz  = l_z[ti] - sh_z[i+ti*block_size_x+j];

                        uint64_t pos = j * ((uint64_t)N) + (bx+i+ti*block_size_x);
                        if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                            correlations[pos] = 1;
                        }

                }

            }

        }
        //same as above but with bounds checks for last few blocks
        else {

            //unfortunately there's no better way to do this right now
            //[1, 2, 3, 4, 5, 6, 10, 12, 15]
            #if f_unroll == 2
            #pragma unroll 2
            #elif f_unroll == 3
            #pragma unroll 3
            #elif f_unroll == 4
            #pragma unroll 4
            #elif f_unroll == 5
            #pragma unroll 5
            #elif f_unroll == 6
            #pragma unroll 6
            #elif f_unroll == 10
            #pragma unroll 10
            #elif f_unroll == 12
            #pragma unroll 12
            #elif f_unroll == 15
            #pragma unroll 15
            #endif            
            for (int j=0; j < window_width; j++) {

                for (int ti=0; ti<tile_size_x; ti++) {

                    if (bx+i+ti*block_size_x+j < N) {

                        float diffct = l_ct[ti] - sh_ct[i+ti*block_size_x+j];
                        float diffx  = l_x[ti] - sh_x[i+ti*block_size_x+j];
                        float diffy  = l_y[ti] - sh_y[i+ti*block_size_x+j];
                        float diffz  = l_z[ti] - sh_z[i+ti*block_size_x+j];

                        uint64_t pos = j * ((uint64_t)N) + (bx+i+ti*block_size_x);
                        if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                            correlations[pos] = 1;
                        }

                    }
                }



            }

        }


    }
}




/*
 * This kernel uses warp-shuffle instructions to re-use many of
 * the input values in registers and reduce the pressure on shared memory.
 * However, it does this so drastically that shared memory is hardly needed anymore.
 *
 * Tuning parameters supported are 'block_size_x', 'shem' [0,1], 'read_only' [0,1], 'use_if' [0,1]
 *
 */
__global__ void quadratic_difference_linear_shfl(char *__restrict__ correlations, int N, int sliding_window_width,
        const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z, const float *__restrict__ ct) {

    int tx = threadIdx.x;
    int bx = blockIdx.x * block_size_x;

    #if shmem == 1
    __shared__ float sh_ct[block_size_x + window_width -1];
    __shared__ float sh_x[block_size_x + window_width -1];
    __shared__ float sh_y[block_size_x + window_width -1];
    __shared__ float sh_z[block_size_x + window_width -1];
    #endif

    if (bx+tx < N) {

        //the loading phase
        #if shmem == 1
        for (int k=tx; k < block_size_x+window_width-1; k+=block_size_x) {
            if (bx+k < N) {
                sh_ct[k] = ct[bx+k];
                sh_x[k] = x[bx+k];
                sh_y[k] = y[bx+k];
                sh_z[k] = z[bx+k];
            }
        }
        __syncthreads();
        #endif

        //start of the the computations phase
        #if shmem == 1
        int i = tx;
        float ct_i = sh_ct[i];
        float x_i = sh_x[i];
        float y_i = sh_y[i];
        float z_i = sh_z[i];
        #elif shmem == 0
        int i = bx + tx;
        float ct_i = LDG(ct,i);
        float x_i = LDG(x,i);
        float y_i = LDG(y,i);
        float z_i = LDG(z,i);
        #endif

        //small optimization to eliminate bounds checks for most blocks
        //if (bx+block_size_x+window_width-1 < N) {

            int laneid = tx & (32-1);

            for (int j=0; j < 32-laneid; j++) {

                #if shmem == 1
                float diffct = ct_i - sh_ct[i+j];
                float diffx  = x_i - sh_x[i+j];
                float diffy  = y_i - sh_y[i+j];
                float diffz  = z_i - sh_z[i+j];
                uint64_t pos = j * ((uint64_t)N) + (bx+i);
                #elif shmem == 0
                float diffct = ct_i - LDG(ct,i+j);
                float diffx = x_i - LDG(x,i+j);
                float diffy = y_i - LDG(y,i+j);
                float diffz = z_i - LDG(z,i+j);
                uint64_t pos = j * ((uint64_t)N) + (i);
                #endif

                #if use_if == 1
                if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                    correlations[pos] = 1;
                }
                #elif use_if == 0
                correlations[pos] = (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz);
                #endif

            }

            int j;

                
            #if f_unroll == 2
            #pragma unroll 2
            #elif f_unroll == 4
            #pragma unroll 4
            #endif
            for (j=32; j < window_width-32; j+=32) {

                #if shmem == 1
                float ct_j = sh_ct[i+j];
                float x_j = sh_x[i+j];
                float y_j = sh_y[i+j];
                float z_j = sh_z[i+j];
                #elif shmem == 0
                float ct_j = LDG(ct,i+j);
                float x_j = LDG(x,i+j);
                float y_j = LDG(y,i+j);
                float z_j = LDG(z,i+j);
                #endif

                for (int d=1; d<33; d++) {
                    ct_j = __shfl(ct_j, laneid+1);
                    x_j = __shfl(x_j, laneid+1);
                    y_j = __shfl(y_j, laneid+1);
                    z_j = __shfl(z_j, laneid+1);

                    float diffct = ct_i - ct_j;
                    float diffx  = x_i - x_j;
                    float diffy  = y_i - y_j;
                    float diffz  = z_i - z_j;

                    int c = laneid+d > 31 ? -32 : 0;

                    #if shmem == 1
                    uint64_t pos = (j+d+c) * ((uint64_t)N) + (bx+i);
                    #elif shmem == 0
                    uint64_t pos = (j+d+c) * ((uint64_t)N) + (i);
                    #endif

                    #if use_if == 1
                    if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                        correlations[pos] = 1;
                    }
                    #elif use_if == 0
                    correlations[pos] = (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz);
                    #endif

                }

            }


            j-=laneid;
            for (; j < window_width; j++) {

                #if shmem == 1
                float diffct = ct_i - sh_ct[i+j];
                float diffx  = x_i - sh_x[i+j];
                float diffy  = y_i - sh_y[i+j];
                float diffz  = z_i - sh_z[i+j];
                uint64_t pos = j * ((uint64_t)N) + (bx+i);
                #elif shmem == 0
                float diffct = ct_i - LDG(ct,i+j);
                float diffx = x_i - LDG(x,i+j);
                float diffy = y_i - LDG(y,i+j);
                float diffz = z_i - LDG(z,i+j);
                uint64_t pos = j * ((uint64_t)N) + (i);
                #endif

                #if use_if == 1
                if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                    correlations[pos] = 1;
                }
                #elif use_if == 0
                correlations[pos] = (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz);
                #endif


            }


/*
        }
        //same as above but with bounds checks for last few blocks
        else {

            for (int j=0; j < window_width; j++) {

                if (bx+i+j >= N) { return; }

                float diffct = ct_i - sh_ct[i+j];
                float diffx  = x_i - sh_x[i+j];
                float diffy  = y_i - sh_y[i+j];
                float diffz  = z_i - sh_z[i+j];

                uint64_t pos = j * ((uint64_t)N) + (bx+i);
                if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                    correlations[pos] = 1;
                }

            }

        }
*/
    }
}





















__global__ void quadratic_difference_linear_clean(char *correlations, int N, int sliding_window_width, float *x, float *y, float *z, float *ct) {

    int tx = threadIdx.x;
    int bx = blockIdx.x * block_size_x;

    __shared__ float sh_ct[block_size_x + window_width -1];
    __shared__ float sh_x[block_size_x + window_width -1];
    __shared__ float sh_y[block_size_x + window_width -1];
    __shared__ float sh_z[block_size_x + window_width -1];

    if (bx+tx < N) {

        //the loading phase
        for (int k=tx; k < block_size_x+window_width-1; k+=block_size_x) {
            if (bx+k < N) {
                sh_ct[k] = ct[bx+k];
                sh_x[k] = x[bx+k];
                sh_y[k] = y[bx+k];
                sh_z[k] = z[bx+k];
            }
        }
        __syncthreads();

        //start of the the computations phase
        int i = tx;
        float ct_i = sh_ct[i];
        float x_i = sh_x[i];
        float y_i = sh_y[i];
        float z_i = sh_z[i];

        //small optimization to eliminate bounds checks for most blocks
        if (bx+block_size_x+window_width-1 < N) {

            //unfortunately there's no better way to do this right now
            //[1, 2, 3, 4, 5, 6, 10, 12, 15]
            #if f_unroll == 2
            #pragma unroll 2
            #elif f_unroll == 3
            #pragma unroll 3
            #elif f_unroll == 4
            #pragma unroll 4
            #elif f_unroll == 5
            #pragma unroll 5
            #elif f_unroll == 6
            #pragma unroll 6
            #elif f_unroll == 10
            #pragma unroll 10
            #elif f_unroll == 12
            #pragma unroll 12
            #elif f_unroll == 15
            #pragma unroll 15
            #endif            
            for (int j=0; j < window_width; j++) {
                float diffct = ct_i - sh_ct[i+j];
                float diffx  = x_i - sh_x[i+j];
                float diffy  = y_i - sh_y[i+j];
                float diffz  = z_i - sh_z[i+j];

                uint64_t pos = j * ((uint64_t)N) + (bx+i);
                if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                    correlations[pos] = 1;
                }

            }

        }
        //same as above but with bounds checks for last few blocks
        else {

            //unfortunately there's no better way to do this right now
            //[1, 2, 3, 4, 5, 6, 10, 12, 15]
            #if f_unroll == 2
            #pragma unroll 2
            #elif f_unroll == 3
            #pragma unroll 3
            #elif f_unroll == 4
            #pragma unroll 4
            #elif f_unroll == 5
            #pragma unroll 5
            #elif f_unroll == 6
            #pragma unroll 6
            #elif f_unroll == 10
            #pragma unroll 10
            #elif f_unroll == 12
            #pragma unroll 12
            #elif f_unroll == 15
            #pragma unroll 15
            #endif            
            for (int j=0; j < window_width; j++) {

                if (bx+i+j >= N) { return; }

                float diffct = ct_i - sh_ct[i+j];
                float diffx  = x_i - sh_x[i+j];
                float diffy  = y_i - sh_y[i+j];
                float diffz  = z_i - sh_z[i+j];

                uint64_t pos = j * ((uint64_t)N) + (bx+i);
                if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                    correlations[pos] = 1;
                }

            }

        }

    }
}




























#ifndef vector_size
 #define vector_size 1
#endif

#if vector_size == 1
#define floatvec float
#elif vector_size == 2 
#define floatvec float2
#elif vector_size == 4
#define floatvec float4
#endif

__device__ __inline__ void load_and_store_vectorized(float *out, int out_index, floatvec *in, int in_index) {
    floatvec f_in = in[in_index];
    #if vector_size == 1
    out[out_index] = f_in;
    #elif vector_size == 2
    out[out_index*2] = f_in.x;
    out[out_index*2+1] = f_in.y;
    #elif vector_size == 4
    out[out_index*4+0] = f_in.x;
    out[out_index*4+1] = f_in.y;
    out[out_index*4+2] = f_in.z;
    out[out_index*4+3] = f_in.w;
    #endif
} 

__global__ void quadratic_difference_linear_experimental(char *correlations, int N, int sliding_window_width, floatvec *x, floatvec *y, floatvec *z, floatvec *ct) {

    int tx = threadIdx.x;
    int bx = blockIdx.x * block_size_x;

    __shared__ float sh_ct[block_size_x + window_width -1];
    __shared__ float sh_x[block_size_x + window_width -1];
    __shared__ float sh_y[block_size_x + window_width -1];
    __shared__ float sh_z[block_size_x + window_width -1];

    if (bx+tx < N) {

        #if vector_size == 1
        int kend = block_size_x+window_width-1;
        #else
        int kend = (block_size_x+window_width-1)/vector_size;
        #endif

        for (int k=tx; k < kend; k+=block_size_x) {
            #if vector_size == 1
            if (bx+k < N) {
                sh_ct[k] = ct[bx+k];
                sh_x[k] = x[bx+k];
                sh_y[k] = y[bx+k];
                sh_z[k] = z[bx+k];
            }
            #else
            int index = (blockIdx.x * block_size_x)/vector_size + k;
            if (index*vector_size < N) {
                load_and_store_vectorized(sh_ct, k, ct, index);
                load_and_store_vectorized(sh_x, k, x, index);
                load_and_store_vectorized(sh_y, k, y, index);
                load_and_store_vectorized(sh_z, k, z, index);
            }
            #endif

        }



        __syncthreads();

        int i = tx;

        float ct_i = sh_ct[i];
        float x_i = sh_x[i];
        float y_i = sh_y[i];
        float z_i = sh_z[i];

        if (bx+block_size_x+window_width-1 < N) {

            for (int j=0; j < window_width; j++) {

                float diffct = ct_i - sh_ct[i+j];
                float diffx  = x_i - sh_x[i+j];
                float diffy  = y_i - sh_y[i+j];
                float diffz  = z_i - sh_z[i+j];

                uint64_t pos = j * ((uint64_t)N) + (bx+i);
                #if (use_if == 1)
                if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                    correlations[pos] = 1;
                }
                #elif (use_if == 0)
                correlations[pos] = (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz);
                #endif

            }

        }
        else {

            for (int j=0; j < window_width && bx+i+j < N; j++) {

                float diffct = ct_i - sh_ct[i+j];
                float diffx  = x_i - sh_x[i+j];
                float diffy  = y_i - sh_y[i+j];
                float diffz  = z_i - sh_z[i+j];

                uint64_t pos = j * ((uint64_t)N) + (bx+i);
                #if (use_if == 1)
                if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                    correlations[pos] = 1;
                }
                #elif (use_if == 0)
                correlations[pos] = (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz);
                #endif

            }

        }

    }
}








