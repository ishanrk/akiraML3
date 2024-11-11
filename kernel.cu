#pragma once
#include "kernel.cuh"
#include<random>
#include<algorithm>
#include<iostream>

#include <device_functions.h>


void random_init(float* data, int dim1)
{
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Fill `data` with random values between 0 and 1
    for (int i = 0; i < dim1; ++i) {
        data[i] = dis(gen);
    }
}

__global__ void vectorAddUM(float* c, float* a, float* b, int dim1)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < dim1)
    {
        c[i] = a[i] + b[i];
        printf("c[%d] = %f\n", i, c[i]);
    }
}
// Helper function for using CUDA to add vectors in parallel.
float* addWithCuda(float* c, float* a, float* b, int dim1)
{
    int id = cudaGetDevice(&id);
    int THRDSZ= 5;
    int BLOCKSZ = (int)ceil(dim1 / THRDSZ);
    float* tempa; float* tempb; float* tempc;
    cudaMalloc(&tempa, dim1 * sizeof(float));

    cudaMalloc(&tempb, dim1 * sizeof(float));

    cudaMalloc(&tempc, dim1 * sizeof(float));

    cudaMemcpy(tempa, a, dim1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(tempb, b, dim1 * sizeof(float), cudaMemcpyHostToDevice);


    vectorAddUM <<<BLOCKSZ, THRDSZ >>> (tempc, tempa, tempb, dim1);

    cudaMemcpy(c, tempc, dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    
    return c;

}

__global__ void multiplyVectorsKernel(float* A, float* B, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        result[idx] = A[idx] * B[idx]; // Multiply corresponding components
    }
}
__global__ void sumVectorKernel(float* result, float* sumResult, int N) {
    __shared__ float partialSum[256]; // Shared memory for partial sums

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadID = threadIdx.x;

    if (idx < N) {
        partialSum[threadID] = result[idx]; // Load values into shared memory
    }
    else {
        partialSum[threadID] = 0.0f; // Handle out-of-bound indices
    }

    __syncthreads();

    // Reduction within the block to sum the values
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadID < stride) {
            partialSum[threadID] += partialSum[threadID + stride];
        }
        __syncthreads();
    }

    // The first thread of the block writes the sum to the global result array
    if (threadID == 0) {
        sumResult[blockIdx.x] = partialSum[0];
    }
}


float dotCUDA(float* vec1, float* vec2, int N) {
    float* d_vec1, * d_vec2, * d_result, * d_sumResult;
    float sum = 0.0f;

    // Allocate memory on device
    cudaMalloc(&d_vec1, N * sizeof(float));
    cudaMalloc(&d_vec2, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));
    int numBlocks = (N + 255) / 256;
    cudaMalloc(&d_sumResult, numBlocks * sizeof(float));  // Partial results for block-wise sum

    // Copy data to device
    cudaMemcpy(d_vec1, vec1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, vec2, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to multiply vectors
    multiplyVectorsKernel << <numBlocks, 256 >> > (d_vec1, d_vec2, d_result, N);

    // Launch kernel to sum the result vector
    sumVectorKernel << <numBlocks, 256 >> > (d_result, d_sumResult, N);

    // Copy partial results back to host
    float* partialResults = new float[numBlocks];
    cudaMemcpy(partialResults, d_sumResult, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum the partial results to get the final sum
    for (int i = 0; i < numBlocks; ++i) {
        sum += partialResults[i];
    }

    // Clean up
    delete[] partialResults;
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_result);
    cudaFree(d_sumResult);

    return sum;
}

__global__ void matrixVectorMulKernel(float* A, float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process rows within bounds
    if (row < M) {
        float sum = 0.0f;
        // Compute the dot product of row `row` of A and vector x
        for (int col = 0; col < N; col++) {
            sum += A[row * N + col] * x[col];
        }
        // Store the result in the output vector y
        y[row] = sum;
    }
}
void matrixVectorMul(float* A, float* x, float* y, int M, int N) {
    float* d_A, * d_x, * d_y;

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, M * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    int blockSize = 256;
    int numBlocks = (M + blockSize - 1) / blockSize;

    // Launch the kernel
    matrixVectorMulKernel << <numBlocks, blockSize >> > (d_A, d_x, d_y, M, N);

    // Copy the result back to host
    cudaMemcpy(y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}


