#pragma once
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
// Function declaration
void random_init(float* data, int dim1, int dim2);
__global__ void vectorAddUM(float* c, float* a, float* b, int dim1);
float* addWithCuda(float* c, float* a, float* b, int dim1);
float dotCUDA(float* vec1, float* vec2, int n);
__global__ void multiplyVectorsKernel(float* A, float* B, float* result, int N);
__global__ void sumVectorKernel(float* result, float* sumResult, int N);
void matrixVectorMul(float* A, float* x, float* y, int M, int N);
__global__ void matrixVectorMulKernel(float* A, float* x, float* y, int M, int N);
void applyReLU(float* input, float* output, int N);
void applySigmoid(float* input, float* output, int N);
__global__ void sigmoidKernel(float* input, float* output, int N);
__global__ void reluKernel(float* input, float* output, int N);

void applySoftmax(float* input, float* output, int N);
__global__ void softmaxKernel(float* input, float* output, int N);
__global__ void sigmoidGradientKernel(const float* x, float* grad, int N);
void sigmoidGradient(const float* x, float* grad, int N);
__global__ void softmaxGradientKernel(const float* x, float* grad, int N);
void softmaxGradient(const float* x, float* grad, int N);
void reluGradient(const float* x, float* grad, int N);
__global__ void reluGradientKernel(const float* x, float* grad, int N);
void rowMatrixMul(float* row, float* matrix, float* result, int n, int m);
__global__ void rowMatrixMulKernel(float* row, float* matrix, float* result, int n, int m);