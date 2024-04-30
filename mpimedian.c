#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#include "pgmio.h"

double start_time_median, end_time_median;
double start_time_total, end_time_total;

// Set image size
#define M 225
#define N 225

int compare(const void *a, const void *b) {
    return (*(float *)a - *(float *)b);
}

int main(int argc, char **argv) {

    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int P = size;
    int MP = M / P;
    int NP = N;

    float image[MP + 2][NP + 2];

    float masterbuf[M][N];
    float buf[MP][NP];

    int i, j;
    char *filename;

    if (rank == 0) {
        start_time_total = MPI_Wtime();
        char input[] = "image225x225.pgm";
        filename = input;
        pgmread(filename, masterbuf, M, N);

        printf("width: %d \nheight: %d\nprocessors: %d\n", M, N, P);
    }

    start_time_median = MPI_Wtime();
    MPI_Scatter(masterbuf, MP * NP, MPI_FLOAT, buf, MP * NP, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (i = 1; i < MP + 1; i++) {
        for (j = 1; j < NP + 1; j++) {
            // Apply Median filter
            float neighborhood[9];
            int index = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    neighborhood[index++] = buf[i + k][j + l];
                }
            }

            // Sort the neighborhood array
            qsort(neighborhood, 9, sizeof(float), compare);

            // Select the middle value as the new pixel value
            image[i][j] = neighborhood[4];
        }
    }
    end_time_median = MPI_Wtime();

    if (rank == 0) {
        printf("Finished");
    }

    for (i = 1; i < MP + 1; i++) {
        for (j = 1; j < NP + 1; j++) {
            buf[i - 1][j - 1] = image[i][j];
        }
    }

    MPI_Gather(buf, MP * NP, MPI_FLOAT, masterbuf, MP * NP, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char output[] = "image-output225x225_median.pgm";
        filename = output;

        printf("\nOutput: <%s>\n", filename);
        pgmwrite(filename, masterbuf, M, N);
        end_time_total = MPI_Wtime();

        double total = (end_time_median - start_time_median);
        printf("Total Parallel Time: %fs\n", total);
        printf("Total Serial Time: %fs\n", (end_time_total - start_time_total) - total);
        printf("Total Time: %fs\n", end_time_total - start_time_total);
    }

    MPI_Finalize();

    return 0;
}