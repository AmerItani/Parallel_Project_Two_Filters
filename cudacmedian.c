%%cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include </content/pgmio.h>

// image dimensions WIDTH & HEIGHT
#define WIDTH 225
#define HEIGHT 225

// Block width WIDTH & HEIGHT
#define BLOCK_W 16
#define BLOCK_H 16

// buffer to read image into
float image[HEIGHT][WIDTH];

// buffer for resulting image
float final[HEIGHT][WIDTH];

// prototype declarations
void load_image();
void call_kernel();
void save_image();

#define MAXLINE 128

float total, median;
cudaEvent_t start_total, stop_total;
cudaEvent_t start_median, stop_median;

__global__ void imageBlur(float *input, float *output, int width, int height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    int numcols = WIDTH;

    float blur;

    if (row < height && col < width && row > 0 && col > 0) {
        int x1, x3, x4, x5, x7;

        x1 = input[(row + 1) * numcols + col];
        x3 = input[row * numcols + (col - 1)];
        x4 = input[row * numcols + col];
        x5 = input[row * numcols + (col + 1)];
        x7 = input[(row - 1) * numcols + col];

        blur = (x1 * 0.2) + (x3 * 0.2) + (x4 * 0.2) + (x5 * 0.2) + (x7 * 0.2);

        output[row * numcols + col] = blur;
    }
}

__global__ void medianFilter(float *input, float *output, int width, int height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    int numcols = WIDTH;

    float blur;

    if (row < height && col < width && row > 0 && col > 0) {
        int x0, x1, x2, x3, x4, x5, x6, x7, x8;

        x0 = input[(row - 1) * numcols + (col - 1)];
        x1 = input[(row - 1) * numcols + col];
        x2 = input[(row - 1) * numcols + (col + 1)];
        x3 = input[row * numcols + (col - 1)];
        x4 = input[row * numcols + col];
        x5 = input[row * numcols + (col + 1)];
        x6 = input[(row + 1) * numcols + (col - 1)];
        x7 = input[(row + 1) * numcols + col];
        x8 = input[(row + 1) * numcols + (col + 1)];

        blur = (x0 + 2 * x1 + x2 + 2 * x3 + 4 * x4 + 2 * x5 + x6 + 2 * x7 + x8) / 16.0;



        output[row * numcols + col] = blur;
    }
}

void load_image() {
    pgmread("/content/image225x225.pgm", (void *)image, WIDTH, HEIGHT);
}

void save_image() {
    pgmwrite("/content/image-outputl225x225.pgm", (void *)final, WIDTH, HEIGHT);
}

void call_kernel() {
    int x, y;
    float *d_input, *d_output;

    printf("Block size: %dx%d\n", BLOCK_W, BLOCK_H);

    size_t memSize = WIDTH * HEIGHT * sizeof(float);

    cudaMalloc(&d_input, memSize);
    cudaMalloc(&d_output, memSize);

    for (y = 0; y < HEIGHT; y++) {
        for (x = 0; x < WIDTH; x++) {
            final[x][y] = 0.0;
        }
    }

    printf("Blocks per grid (width): %d |", (WIDTH / BLOCK_W));
    printf("Blocks per grid (height): %d |", (HEIGHT / BLOCK_H));

    cudaMemcpy(d_input, image, memSize, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_W, BLOCK_H);
    dim3 blocks(WIDTH / BLOCK_W, HEIGHT / BLOCK_H);

    imageBlur<<>>(d_input, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    cudaMemcpy(d_input, d_output, memSize, cudaMemcpyDeviceToHost);

    medianFilter<<>>(d_input, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    cudaMemcpy(final, d_output, memSize, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", "Main Loop", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char *argv[]) {
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);

    cudaEventCreate(&start_median);
    cudaEventCreate(&stop_median);

    cudaEventRecord(start_total, 0);

    load_image();

    cudaEventRecord(start_median, 0);

    call_kernel();

    cudaEventRecord(stop_median, 0);
    cudaEventSynchronize(stop_median);
    cudaEventElapsedTime(&median, start_median, stop_median);

    save_image();

    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&total, start_total, stop_total);

    printf("Total Parallel Time: %f s |", median / 1000);
    printf("Total Serial Time: %f s |", (total - median) / 1000);
    printf("Total Time: %f s |", total / 1000);

    cudaDeviceReset();

    return 0;
}