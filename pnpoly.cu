#define VERTICES 600

__constant__ float2 d_Vertices[VERTICES];

//tuning parameters
//block_size_x any sensible thread block size
//tile_size any sensible tile size value
//prefetch 0 or 1 for reusing constant memory from previous iteration

__global__ void cn_PnPoly(int* bitmap, float2* points, int n) {
    int ti = blockIdx.x * block_size_x * tile_size + threadIdx.x;

    if (ti < n) {

        // the crossing number counter
        int cn[tile_size];
        float2 p[tile_size];
        #pragma unroll
        for (int k=0; k<tile_size; k++) {
            cn[k] = 0;
            p[k] = points[ti+k*block_size_x];
        }

        #if prefetch == 1
        float2 vp = d_Vertices[0];
        #endif

        // loop through all edges of the polygon
        for (int i=0; i<VERTICES-1; i++) {    // edge from v to vp

            #if prefetch == 1
            float2 v = vp;
            vp = d_Vertices[i+1];
            #else
            float2 v = d_Vertices[i];
            float2 vp = d_Vertices[i+1];
            #endif
            float vb = (vp.x - v.x) / (vp.y - v.y);

            #pragma unroll
            for (int k=0; k<tile_size; k++) {

                int b = ((v.y <= p[k].y) && (vp.y > p[k].y)) || ((v.y > p[k].y) && (vp.y <= p[k].y));
                cn[k] += b && (p[k].x < v.x + vb * (p[k].y - v.y));

            }
        }


        #if use_bitmap == 1
        int lane_index = threadIdx.x & (32 - 1);
        unsigned int bitstring[tile_size];
        #if coalesce_bitmap == 1
        __shared__ unsigned int block_output[tile_size*block_size_x/32];
        int warp_id = threadIdx.x/32;
        #endif

        #pragma unroll
        for (int k=0; k<tile_size; k++) {
            //write at my position in bitstring
            bitstring[k] = (cn[k] & 1) << (32-lane_index);
            //compute sum of bitstring within warp
            #pragma unroll
            for (unsigned int s=16; s>0; s>>=1) {
                bitstring[k] += __shfl_xor(bitstring[k], s);
            }

            #if coalesce_bitmap == 1
            //store bitstring for this warp in shared buffer
            if (lane_index == 0) {
                block_output[warp_id+k*block_size_x/32] = bitstring[k];
            }
            #endif
        }
        __syncthreads();

        #endif

        #pragma unroll
        for (int k=0; k<tile_size; k++) {

            #if use_bitmap == 0
            bitmap[ti+k*block_size_x] = (cn[k] & 1); // 0 if even (out), and 1 if odd (in)

            #elif use_bitmap == 1
                #if coalesce_bitmap == 0
                if (lane_index == 0) {
                    bitmap[ti/32+k*block_size_x/32] = bitstring[k];
                }
                #elif coalesce_bitmap == 1
                //write back results in coalesced manner
                if (threadIdx.x < block_size_x/32) {
                    bitmap[ti/32+k*block_size_x/32] = block_output[warp_id];
                }
                #endif

            #elif use_bitmap == 2
            if (cn[k] & 1 == 1) {
                bitmap[ti+k*block_size_x] = 1; // 0 if even (out), and 1 if odd (in)
            }

            #endif
        }
    }
}




__global__ void cn_PnPoly_naive(int* bitmap, float2* points, int n) {
    int ti = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    if (ti < n) {

        // the crossing number counter
        int c = 0;
        float2 p = points[ti];

        // loop through all edges of the polygon
        for (int i=0; i<VERTICES-1; i++) {    // edge from v to vp
            float2 v = d_Vertices[i];
            float2 vp = d_Vertices[i+1];


            if ( ((v.y>p.y) != (vp.y>p.y)) && (p.x < (vp.x-v.x) * (p.y-v.y) / (vp.y-v.y) + v.x) ) {
                //c = !c;
                c += 1;
            }


            //float vb = (vp.x - v.x) / (vp.y - v.y);

            //int b = ((v.y <= p.y) && (vp.y > p.y)) || ((v.y > p.y) && (vp.y <= p.y));
            //cn += b && (p.x < v.x + vb * (p.y - v.y));

        }

        //bitmap[ti] = (cn & 1); // 0 if even (out), and 1 if odd (in)
//        bitmap[ti] = c; // 0 if even (out), and 1 if odd (in)
        bitmap[ti] = (int)p.x; // 0 if even (out), and 1 if odd (in)

    }
}

