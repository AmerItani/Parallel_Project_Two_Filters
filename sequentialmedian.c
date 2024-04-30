#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "pgmio.h"

#define M 225
#define N 225
#define THRESH 100

int compare(const void *a, const void *b) {
    return (*(float *)a - *(float *)b);
}

int main(int argc, char **argv)
{
    int i, j, k, l;
    float masterbuf[M][N];

    char *filename;
    char input[] = "image225x225.pgm";
    char output[] = "image-output225x225.pgm";
    filename = input;

    clock_t start_time_total, end_time_total;
    clock_t start_time_median, end_time_median;

    start_time_total = clock();  // For total time measurement
    pgmread(filename, masterbuf, M, N);

    printf("width: %d \nheight: %d\n", M, N);

    float image[M][N];

    start_time_median = clock();  // For median filter time measurement

    for (i = 1; i < M - 1; i++) {
        for (j = 1; j < N - 1; j++) {
            float neighborhood[9];
            int index = 0;
            for (k = -1; k <= 1; k++) {
                for (l = -1; l <= 1; l++) {
                    neighborhood[index++] = masterbuf[i + k][j + l];
                }
            }
            // Sort the neighborhood array
            qsort(neighborhood, 9, sizeof(float), compare);

            // Get the median value
            image[i][j] = neighborhood[4];
        }
    }

    end_time_median = clock();

    printf("Finished\n");

    filename = output;
    printf("Output: <%s>\n", filename);
    pgmwrite(filename, image, M, N);

    end_time_total = clock();

    double median_time = ((double)(end_time_median - start_time_median)) / CLOCKS_PER_SEC;
    double total_time = ((double)(end_time_total - start_time_total)) / CLOCKS_PER_SEC;
    
    printf("Median Filter Time: %fs\n", median_time);
    printf("Total Time: %fs\n", total_time);

    return 0;
}