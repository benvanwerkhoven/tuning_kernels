#include <inttypes.h>

#ifndef block_size_x
#define block_size_x 256
#endif

#define window_width 1500
//#define window_width 32



__global__ void dense2sparse_kernel(int *row_idx, int *col_idx, int *prefix_sums, uint8_t *correlations, int n) {
    int i = blockIdx.x * block_size_x + threadIdx.x;

    if (i<n) {

        int offset = 0;
        if (i>0) {
            offset = prefix_sums[i-1];
        }

        //could do some pruning here looking up prefix_sums[i+1]
        //and see if there is actually any work on this row

        for (int j=0; j<window_width; j++) {

            uint64_t pos = (j * (uint64_t) n) + (uint64_t)i;
            if (correlations[pos] == 1) {
                row_idx[offset] = i;
                col_idx[offset] = j;
                offset += 1;
            }

        }



    }
}
