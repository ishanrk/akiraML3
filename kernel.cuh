#pragma once
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
// Function declaration
void random_init(float* data, int dim1);
__global__ void vectorAddUM(float* c, float* a, float* b, int dim1);
float* addWithCuda(float* c, float* a, float* b, int dim1);
float dotCUDA(float* vec1, float* vec2, int n);
__global__ void multiplyVectorsKernel(float* A, float* B, float* result, int N);
__global__ void sumVectorKernel(float* result, float* sumResult, int N);
void matrixVectorMul(float* A, float* x, float* y, int M, int N);
__global__ void matrixVectorMulKernel(float* A, float* x, float* y, int M, int N);