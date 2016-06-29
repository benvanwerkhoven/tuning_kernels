#ifndef block_size_x
#define block_size_x 16
#endif
#ifndef block_size_y
#define block_size_y 16
#endif

#ifndef tile_size_x
#define tile_size_x 1
#endif
#ifndef tile_size_y
#define tile_size_y 1
#endif


__global__ void transpose_naive(float *output, float *input, int width, int height) {
    int x = blockIdx.x * block_size_x + tx;
    int y = blockIdx.y * block_size_y + ty;

    if (x < width && y < height) {
        output[x*height + y] = input[y*width + x];
    }
}


//for the shared kernel block_size_x has to be equal to block_size_y
__global__ void transpose_shared(float *output, float *input, int width, int height) {
    __shared__ float tile[block_size_x][block_size_y];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * block_size_x + tx;
    int y = blockIdx.y * block_size_y + ty;

    if (x < width && y < height) {
        tile[ty][tx] = input[y*width + x];

        output[x*height + y] = tile[tx][ty];
    }
}


__global__ void transpose_kernel(float *output, float *input, int width, int height) {
    __shared__ float tile[block_size_x*tile_size_x][block_size_y*tile_size_y];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * block_size_x * tile_size_x + tx;
    int y = blockIdx.y * block_size_y * tile_size_y + ty;

    for (int j=0; j<tile_size_y; j++) {
        for (int i=0; i<tile_size_x; i++) {

            int yi = y+j*block_size_y;
            int xi = x+i*block_size_x;
            if (xi < width && yi < height) {
                tile[ty][tx] = input[yi*width + xi];
            }
        }
    }

    for (int j=0; j<tile_size_y; j++) {
        for (int i=0; i<tile_size_x; i++) {

            int yi = y+j*block_size_y;
            int xi = x+i*block_size_x;
            if (xi < width && yi < height) {
                output[xi*height + yi] = tile[tx][ty];
            }
        }
    }
}

