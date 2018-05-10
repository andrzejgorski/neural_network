#include <time.h>
#include <stdio.h>
#include "matrix_mult.h"


static void handleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))


__constant__ int cuda_matrix_sizes[3];
/*
    size[0] = first_r:
    size[1] = second_c:
    size[2] = first_c:
*/


void print_matrix(double *matrix, int rows, int columns) {

    double (*matrix_ptr)[rows][columns] = (
        double(*)[rows][columns]) matrix;

    printf("printing matrix:\n");
    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < columns; j ++) {
            printf("%f ", (*matrix_ptr)[i][j]);
        }
        printf("\n");
    }
}


void matrix_mult(
    double *first, double *second, double *out,
    int first_r, int second_c, int first_c)
{
    double (*first_ptr)[first_r][first_c] = (
        double(*)[first_r][first_c]) first;
    double (*second_ptr)[first_c][second_c] = (
        double(*)[first_c][second_c]) second;
    double (*out_ptr)[first_r][second_c] = (
        double(*)[first_r][second_c]) out;

    for (int b = 0; b < first_r; b++) {
        for (int c = 0; c < second_c; c++) {
            (*out_ptr)[b][c] = 0;
            for (int a = 0; a < first_c; a++) {
                (*out_ptr)[b][c] += (*first_ptr)[b][a] * (*second_ptr)[a][c];
            }
        }
    }
}


void matrix_mult_tf_to(
    double *first, double *second, double *out,
    int first_r, int second_c, int first_c)
{
    double (*first_ptr)[first_c][first_r] = (
        double(*)[first_c][first_r]) first;
    double (*second_ptr)[first_c][second_c] = (
        double(*)[first_c][second_c]) second;
    double (*out_ptr)[second_c][first_r] = (
        double(*)[second_c][first_r]) out;

    for (int b = 0; b < first_r; b++) {
        for (int c = 0; c < second_c; c++) {
            (*out_ptr)[c][b] = 0;
            for (int a = 0; a < first_c; a++) {
                (*out_ptr)[c][b] += (*first_ptr)[a][b] * (*second_ptr)[a][c];
            }
        }
    }
}


void matrix_mult_ts(
    double *first, double *second, double *out,
    int first_r, int second_c, int first_c)
{
    double (*first_ptr)[first_r][first_c] = (
        double(*)[first_r][first_c]) first;
    double (*second_ptr)[second_c][first_c] = (
        double(*)[second_c][first_c]) second;
    double (*out_ptr)[first_r][second_c] = (
        double(*)[first_r][second_c]) out;

    for (int b = 0; b < first_r; b++) {
        for (int c = 0; c < second_c; c++) {
            (*out_ptr)[b][c] = 0;
            for (int a = 0; a < first_c; a++) {
                (*out_ptr)[b][c] += (*first_ptr)[b][a] * (*second_ptr)[c][a];
            }
        }
    }
}


__global__ void cell_gpu_matrix_mult(double *first, double *second, double *out)
{
    int first_r = cuda_matrix_sizes[0];
    int second_c = cuda_matrix_sizes[1];
    int first_c = cuda_matrix_sizes[2];

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;

    int b = tidx / second_c;
    int c = tidx % second_c;
    if (b < first_r && c < second_c) {
        double *out_ptr = out + b * second_c + c;
        *out_ptr = 0;
        double *first_row = first + b * first_c;
        double *second_col = second + c;
        for (int a = 0; a < first_c; a++) {
            (*out_ptr) += *(first_row + a) * *(second_col + a * second_c);
        }
    }
}


__global__ void cell_gpu_matrix_mult_tf_to(double *first, double *second, double *out)
{
    int first_r = cuda_matrix_sizes[0];
    int second_c = cuda_matrix_sizes[1];
    int first_c = cuda_matrix_sizes[2];

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;

    int b = tidx / second_c;
    int c = tidx % second_c;
    if (b < first_r && c < second_c) {
        double *out_ptr = out + b + c * first_r;
        *out_ptr = 0;
        double *first_row = first + b;
        double *second_col = second + c;
        for (int a = 0; a < first_c; a++) {
            (*out_ptr) += *(first_row + a * first_r) * *(second_col + a * second_c);
        }
    }
}


__global__ void cell_gpu_matrix_mult_ts(double *first, double *second, double *out)
{
    int first_r = cuda_matrix_sizes[0];
    int second_c = cuda_matrix_sizes[1];
    int first_c = cuda_matrix_sizes[2];

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;

    int b = tidx / second_c;
    int c = tidx % second_c;
    if (b < first_r && c < second_c) {
        double *out_ptr = out + b * second_c + c;
        *out_ptr = 0;
        double *first_row = first + b * first_c;
        double *second_col = second + c * first_c;
        for (int a = 0; a < first_c; a++) {
            (*out_ptr) += *(first_row + a) * *(second_col + a);
        }
    }
}


void matrix_mult_gpu (
    double *first, double *second, double *out,
    int first_r, int second_c, int first_c)
{
    double *cuda_first, *cuda_second, *cuda_out;

    int *cpu_sizes = (int*) malloc(sizeof(int) * 3);
    cpu_sizes[0] = first_r;
    cpu_sizes[1] = second_c;
    cpu_sizes[2] = first_c;

    int first_elems = first_r * first_c;
    int second_elems = first_c * second_c;
    int out_elems = first_r * second_c;

    cudaMalloc((void**) &cuda_first, first_elems * sizeof(double));
    cudaMalloc((void**) &cuda_second, second_elems * sizeof(double));
    cudaMalloc((void**) &cuda_out, out_elems * sizeof(double));

    cudaMemcpy(cuda_first, first, first_elems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_second, second, second_elems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_out, out, out_elems * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(cuda_matrix_sizes, cpu_sizes, sizeof(int) * 3);

    int blocksize = 1024;
    int blocknum = (out_elems + blocksize - 1) / blocksize;

    // puts("before kernel");
    cell_gpu_matrix_mult<<<blocknum, blocksize>>>(
        cuda_first, cuda_second, cuda_out);

    cudaCheck(cudaPeekAtLastError());

    cudaMemcpy(out, cuda_out, out_elems * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(cuda_first);
    cudaFree(cuda_second);
    cudaFree(cuda_out);
}


void matrix_mult_gpu_tf_to (
    double *first, double *second, double *out,
    int first_r, int second_c, int first_c)
{
    double *cuda_first, *cuda_second, *cuda_out;

    int *cpu_sizes = (int*) malloc(sizeof(int) * 3);
    cpu_sizes[0] = first_r;
    cpu_sizes[1] = second_c;
    cpu_sizes[2] = first_c;

    int first_elems = first_r * first_c;
    int second_elems = first_c * second_c;
    int out_elems = first_r * second_c;

    cudaMalloc((void**) &cuda_first, first_elems * sizeof(double));
    cudaMalloc((void**) &cuda_second, second_elems * sizeof(double));
    cudaMalloc((void**) &cuda_out, out_elems * sizeof(double));

    cudaMemcpy(cuda_first, first, first_elems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_second, second, second_elems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_out, out, out_elems * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(cuda_matrix_sizes, cpu_sizes, sizeof(int) * 3);

    int blocksize = 1024;
    int blocknum = (out_elems + blocksize - 1) / blocksize;

    // puts("before kernel");
    cell_gpu_matrix_mult_tf_to<<<blocknum, blocksize>>>(
        cuda_first, cuda_second, cuda_out);

    cudaCheck(cudaPeekAtLastError());

    cudaMemcpy(out, cuda_out, out_elems * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(cuda_first);
    cudaFree(cuda_second);
    cudaFree(cuda_out);
}


void matrix_mult_gpu_ts (
    double *first, double *second, double *out,
    int first_r, int second_c, int first_c)
{
    double *cuda_first, *cuda_second, *cuda_out;

    int *cpu_sizes = (int*) malloc(sizeof(int) * 3);
    cpu_sizes[0] = first_r;
    cpu_sizes[1] = second_c;
    cpu_sizes[2] = first_c;

    int first_elems = first_r * first_c;
    int second_elems = first_c * second_c;
    int out_elems = first_r * second_c;

    cudaMalloc((void**) &cuda_first, first_elems * sizeof(double));
    cudaMalloc((void**) &cuda_second, second_elems * sizeof(double));
    cudaMalloc((void**) &cuda_out, out_elems * sizeof(double));

    cudaMemcpy(cuda_first, first, first_elems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_second, second, second_elems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_out, out, out_elems * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(cuda_matrix_sizes, cpu_sizes, sizeof(int) * 3);

    int blocksize = 1024;
    int blocknum = (out_elems + blocksize - 1) / blocksize;

    // puts("before kernel");
    cell_gpu_matrix_mult_ts<<<blocknum, blocksize>>>(
        cuda_first, cuda_second, cuda_out);

    cudaCheck(cudaPeekAtLastError());

    cudaMemcpy(out, cuda_out, out_elems * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(cuda_first);
    cudaFree(cuda_second);
    cudaFree(cuda_out);
}
