#ifndef CUDA_H
#define CUDA_H

#ifdef GPU

#include "darknet.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cublas_api.h"

void check_error(cudaError_t status);
cublasHandle_t blas_handle();
int *cuda_make_int_array(int *x, size_t n);
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif
#endif
