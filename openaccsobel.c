#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <openacc.h>
#include </content/pgmio.h>

// image dimensions WIDTH & HEIGHT
#define WIDTH 225
#define HEIGHT 225

// buffer to read image into
float image[HEIGHT][WIDTH];

// buffer for resulting image
float final[HEIGHT][WIDTH];

// prototype declarations
void load_image();
void call_kernel();
void save_image();

#define MAXLINE 128

float total, sobel;

void imageBlur(float *input, float *output, int width, int height);
void sobelFilter(float *input, float *output, int width, int height);

void load_image() {
    pgmread("/content/image225x225.pgm", (void *)image, WIDTH, HEIGHT);
}

void save_image() {
    pgmwrite("/content/image-outputl225x225.pgm", (void *)final, WIDTH, HEIGHT);
}

void call_kernel() {
    int x, y;

    printf("Block size: %dx%d\n", BLOCK_W, BLOCK_H);

    size_t memSize = WIDTH * HEIGHT * sizeof(float);

    float *d_input, *d_output;
    d_input = (float *)malloc(memSize);
    d_output = (float *)malloc(memSize);

    for (y = 0; y < HEIGHT; y++) {
        for (x = 0; x < WIDTH; x++) {
            final[x][y] = 0.0;
        }
    }

    printf("Blocks per grid (width): %d |", (WIDTH / BLOCK_W));
    printf("Blocks per grid (height): %d |", (HEIGHT / BLOCK_H));

    #pragma acc data copyin(image) copyout(d_input)
    {
        #pragma acc parallel loop collapse(2)
        for (y = 0; y < HEIGHT; y++) {
            for (x = 0; x < WIDTH; x++) {
                d_input[y * WIDTH + x] = image[y][x];
            }
        }

        #pragma acc data copyin(d_input) copyout(d_output)
        {
            #pragma acc parallel loop collapse(2)
            for (y = 0; y < HEIGHT; y++) {
                for (x = 0; x < WIDTH; x++) {
                    final[x][y] = 0.0;
                }
            }

            dim3 threads(BLOCK_W, BLOCK_H);

            #pragma acc parallel loop gang, vector(threads.x, threads.y)
            for (y = 0; y < HEIGHT; y++) {
                #pragma acc loop vector
                for (x = 0; x < WIDTH; x++) {
                    imageBlur(d_input, d_output, WIDTH, HEIGHT);
                }
            }

            #pragma acc parallel loop gang, vector(threads.x, threads.y)
            for (y = 0; y < HEIGHT; y++) {
                #pragma acc loop vector
                for (x = 0; x < WIDTH; x++) {
                    sobelFilter(d_input, d_output, WIDTH, HEIGHT);
                }
            }
        }
    }

    #pragma acc data copyout(final)
    {
        #pragma acc parallel loop collapse(2)
        for (y = 0; y < HEIGHT; y++) {
            for (x = 0; x < WIDTH; x++) {
                final[x][y] = d_output[y * WIDTH + x];
            }
        }
    }

    free(d_input);
    free(d_output);
}

int main(int argc, char *argv[]) {
    load_image();

    double start_total, stop_total, start_sobel, stop_sobel;

    start_total = omp_get_wtime();

    call_kernel();

    stop_sobel = omp_get_wtime();
    sobel = (stop_sobel - start_sobel) * 1000;

    save_image();

    stop_total = omp_get_wtime();
    total = (stop_total - start_total) * 1000;

    printf("Total Parallel Time:  %f s |", sobel / 1000);
    printf("Total Serial Time:  %f s |", (total - sobel) / 1000);
    printf("Total Time:  %f s |", total / 1000);

    return 0;
}
