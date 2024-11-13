#pragma once
#include <cuda_runtime.h>
#include<cmath>
#include<device_launch_parameters.h>
#include<iostream>
#include<vector>
#include "kernel.cuh"
#include<random>
#include<algorithm>
#include <cmath>
#include <utility>
#include <random>
#include <device_functions.h>
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
__global__ void transposeKernel(float* input, float* output, int width, int height);
void transposeMatrix(float* d_input, float* d_output, int width, int height);
void transposeMatrixCPU(float* input, float* output, int rows, int cols);
void computeRMSEDerivative(float* pred, float* actual, float* grad, int N, float RMSE);
__global__ void rsmeDerivativeKernel(float* pred, float* actual, float* grad, int N, float RMSE);
float computeRMSE(float* pred, float* actual, int N);
__global__ void rsmeKernel(float* pred, float* actual, float* output, int N);
__global__ void elementwiseMultiplyKernel(float* x, float* y, float* result, int N);
void elementwiseMultiply(float* x, float* y, float* result, int N);	
std::vector<std::pair<float, float>> generateLinearData(int num_samples, float slope, float intercept, float noise_stddev);