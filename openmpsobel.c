#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "pgmio.h"

#define M 225
#define N 225
#define THRESH 100

int main(int argc, char **argv)
{
    int i, j;
    float masterbuf[M][N];
    char *filename;
    char input[] = "superman.pgm";
    char output[] = "image-output225x225.pgm";
    filename = input;

    // Set the number of threads
   
    omp_set_num_threads(2);

    double start_time_total, end_time_total;
    double start_time_sobel, end_time_sobel;

    start_time_total = omp_get_wtime();
    pgmread(filename, masterbuf, M, N);

    printf("width: %d \nheight: %d\n", M, N);
    start_time_sobel = omp_get_wtime();

    float image[M][N];

    #pragma omp parallel for private(i, j) shared(masterbuf, image)
    for (i = 1; i < M - 1; i++) {
        for (j = 1; j < N - 1; j++) {
            float gradient_h = ((-1.0 * masterbuf[i - 1][j - 1]) + (1.0 * masterbuf[i + 1][j - 1]) + (-2.0 * masterbuf[i - 1][j]) + (2.0 * masterbuf[i + 1][j]) + (-1.0 * masterbuf[i - 1][j + 1]) + (1.0 * masterbuf[i + 1][j + 1]));
            float gradient_v = ((-1.0 * masterbuf[i - 1][j - 1]) + (-2.0 * masterbuf[i][j - 1]) + (-1.0 * masterbuf[i + 1][j - 1]) + (1.0 * masterbuf[i - 1][j + 1]) + (2.0 * masterbuf[i][j + 1]) + (1.0 * masterbuf[i + 1][j + 1]));
            float gradient = sqrt((gradient_h * gradient_h) + (gradient_v * gradient_v));
            image[i][j] = (gradient < THRESH) ? 0 : 255;
        }
    }

    end_time_sobel = omp_get_wtime();

    printf("Finished\n");

    filename = output;
    printf("Output: <%s>\n", filename);
    pgmwrite(filename, image, M, N);

    end_time_total = omp_get_wtime();

    double total = (end_time_sobel - start_time_sobel);
    printf("Total Parallel Time: %fs\n", total);
    printf("Total Serial Time: %fs\n", (end_time_total - start_time_total) - total);
    printf("Total Time: %fs\n", end_time_total - start_time_total);

    return 0;
}
