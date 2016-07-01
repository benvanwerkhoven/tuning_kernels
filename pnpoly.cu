#define VERTICES 600

__constant__ float2 d_Vertices[VERTICES];

//tuning parameters
//block_size_x any sensible thread block size
//tile_size any sensible tile size value
//prefetch 0 or 1 for reusing constant memory from previous iteration

//#ifndef prefetch
//#define prefetch 0
//#endif

#ifndef use_bitmap
#define use_bitmap 0
#define coalesce_bitmap 0
#endif

#ifndef block_size_x
#define block_size_x 256
#endif

#ifndef tile_size
#define tile_size 1
#endif


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

        int k = VERTICES-1;
//        #if prefetch == 1
//        float2 vj; // = d_Vertices[k];
//        float2 vk;
//        #endif

        // loop through all edges of the polygon
        for (int j=0; j<VERTICES; k = j++) {    // edge from v to vp

//            #if prefetch == 1
//            float2 vj = d_Vertices[j]; 
//            float2 vk = d_Vertices[k]; 
//            #else
            float2 vj = d_Vertices[j]; 
            float2 vk = d_Vertices[k]; 
//            #endif

            #if method == 1
            float vb = (vj.x - vk.x) / (vj.y - vk.y);
            #endif

            #pragma unroll
            for (int i=0; i<tile_size; i++) {

                #if method == 0
                if ( ((vj.y>p[i].y) != (vk.y>p[i].y)) &&
                        (p[i].x < (vk.x-vj.x) * (p[i].y-vj.y) / (vk.y-vj.y) + vj.x) ) {
                    cn[i] = !cn[i];
                }
                #elif method == 1
                int b = ((vk.y <= p[k].y) && (vj.y > p[k].y)) || ((vk.y > p[k].y) && (vj.y <= p[k].y));
                cn[k] += b && (p[k].x < vk.x + vb * (p[k].y - vj.y));

                #endif


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
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
//    int ti = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    if (ti < n) {

        // the crossing number counter
        int c = 0;
        float2 p = points[ti];

        int k = VERTICES-1;

        for (int j=0; j<VERTICES; k = j++) {    // edge from v to vp

            float2 vj = d_Vertices[j]; 
            float2 vk = d_Vertices[k]; 

            if ( ((vj.y>p.y) != (vk.y>p.y)) &&
                    (p.x < (vk.x-vj.x) * (p.y-vj.y) / (vk.y-vj.y) + vj.x) )
                c = !c;

        }

        bitmap[ti] = c; // 0 if even (out), and 1 if odd (in)

    }
}


/*CPU version*/
float pnpoly_cn(int *bitmap, float2 *v, float2 *p) {
    int nvert = VERTICES;
    int npoint = 20000;

    int i = 0;

    for (i = 0; i < npoint; i++) {
        int j, k, c = 0;
        for (j = 0, k = nvert-1; j < nvert; k = j++) {
            if ( ((v[j].y>p[i].y) != (v[k].y>p[i].y)) &&
                    (p[i].x < (v[k].x-v[j].x) * (p[i].y-v[j].y) / (v[k].y-v[j].y) + v[j].x) )
                c = !c;
        }
        bitmap[i] = c & 1;
    }

    return 0.0;
}

/*GPU version*/
__global__ void pnpoly_cn_gpu(int *bitmap, float2 *points, int n) {
    int i = blockIdx.x * block_size_x + threadIdx.x;

    if (i < n) {

        float2 p = points[i];
        int c = 0;
        int k = VERTICES-1;
        int j = 0;

        for (j = 0; j < VERTICES; k = j++) {
            float2 vj = d_Vertices[j]; 
            float2 vk = d_Vertices[k]; 

            if ( ((vj.y>p.y) != (vk.y>p.y)) &&
                    (p.x < (vk.x-vj.x) * (p.y-vj.y) / (vk.y-vj.y) + vj.x) )
                c = !c;

        }

        bitmap[i] = c;

    }
}
