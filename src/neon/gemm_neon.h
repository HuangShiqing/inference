#ifndef GEMM_NEON_H
#define GEMM_NEON_H

void gemm_nn_neon(int M, int N, int K, //float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);

#endif        