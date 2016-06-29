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




__global__ void quadratic_difference_linear(char *correlations, int N, int sliding_window_width, float *x, float *y, float *z, float *ct) {

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

            for (int j=0; j < window_width && bx+i+j < N; j++) {

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








