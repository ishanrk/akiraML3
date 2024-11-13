#pragma once
#include "kernel.cuh"
#include<random>
#include<algorithm>
#include<iostream>

#include <device_functions.h>


void random_init(float* data, int dim1, int dim2)
{
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Fill `data` with random values between 0 and 1
    for (int i = 0; i < dim1*dim2; ++i) {
        data[i] = dis(gen);
    }
}

__global__ void vectorAddUM(float* c, float* a, float* b, int dim1)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < dim1)
    {
        c[i] = a[i] + b[i];
        
    }
}
// Helper function for using CUDA to add vectors in parallel.
float* addWithCuda(float* c, float* a, float* b, int dim1)
{
    int id = cudaGetDevice(&id);
    int THRDSZ= 5;
    int BLOCKSZ = (int)ceil((double)dim1 / (double)THRDSZ);
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


__global__ void sigmoidKernel(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we do not go out of bounds
    if (idx < N) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));  // Sigmoid function
    }
}
void applySigmoid(float* input, float* output, int N) {
    float* d_input, * d_output;
    size_t size = N * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy data from host to device
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Launch the sigmoid kernel
    int blockSize = 256; // number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize; // number of blocks
    sigmoidKernel << <numBlocks, blockSize >> > (d_input, d_output, N);

    // Check for errors
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

void applyReLU(float* input, float* output, int N) {
    float* d_input, * d_output;
    size_t size = N * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy data from host to device
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Launch the ReLU kernel
    int blockSize = 256; // number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize; // number of blocks
    reluKernel << <numBlocks, blockSize >> > (d_input, d_output, N);

    // Check for errors
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}


__global__ void reluKernel(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we do not go out of bounds
    if (idx < N) {
        output[idx] = fmaxf(0.0f, input[idx]);  // ReLU function
    }
}
__global__ void softmaxKernel(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we do not go out of bounds
    if (idx < N) {
        float sumExp = 0.0f;

        // First, compute the sum of the exponentials (this could be optimized if using shared memory)
        for (int i = 0; i < N; i++) {
            sumExp += expf(input[i]);
        }

        // Now compute the softmax for this index
        output[idx] = expf(input[idx]) / sumExp;
    }
}

void applySoftmax(float* input, float* output, int N) {
    float* d_input, * d_output;
    size_t size = N * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy data from host to device
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Launch the softmax kernel
    int blockSize = 256; // number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize; // number of blocks
    softmaxKernel << <numBlocks, blockSize >> > (d_input, d_output, N);

    // Check for errors
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
__global__ void sigmoidGradientKernel(const float* x, float* grad, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the thread is within the bounds of the input vector
    if (idx < N) {
        // Sigmoid of the input element
        float sigmoid_val = 1.0f / (1.0f + expf(-x[idx]));

        // Gradient of sigmoid: sigmoid(x) * (1 - sigmoid(x))
        grad[idx] = sigmoid_val * (1.0f - sigmoid_val);
    }
}

// Function to invoke the CUDA kernel
void sigmoidGradient(const float* x, float* grad, int N) {
    float* d_x, * d_grad;

    // Allocate memory on the device
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_grad, N * sizeof(float));

    // Copy the input data from host to device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block size for CUDA kernel
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize;  // Number of blocks needed

    // Launch the kernel
    sigmoidGradientKernel << <numBlocks, blockSize >> > (d_x, d_grad, N);

    // Copy the result back from device to host
    cudaMemcpy(grad, d_grad, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the allocated memory on the device
    cudaFree(d_x);
    cudaFree(d_grad);
}

__global__ void softmaxGradientKernel(const float* x, float* grad, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we're within bounds of the input vector
    if (idx < N) {
        // First, compute the softmax values for the input vector
        float sum_exp = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum_exp += expf(x[i]);
        }

        // Softmax values
        float softmax_val = expf(x[idx]) / sum_exp;

        // Calculate the gradient for the softmax (Jacobian elements)
        for (int j = 0; j < N; ++j) {
            if (idx == j) {
                grad[idx * N + j] = softmax_val * (1.0f - softmax_val); // Diagonal (i == j)
            }
            else {
                grad[idx * N + j] = -softmax_val * expf(x[j]) / sum_exp; // Off-diagonal (i != j)
            }
        }
    }
}

void softmaxGradient(const float* x, float* grad, int N) {
    float* d_x, * d_grad;

    // Allocate memory on the device
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_grad, N * N * sizeof(float));  // N x N matrix for Jacobian

    // Copy the input data from host to device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block size for CUDA kernel
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize;  // Number of blocks needed

    // Launch the kernel
    softmaxGradientKernel << <numBlocks, blockSize >> > (d_x, d_grad, N);

    // Copy the result back from device to host
    cudaMemcpy(grad, d_grad, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the allocated memory on the device
    cudaFree(d_x);
    cudaFree(d_grad);
}

__global__ void reluGradientKernel(const float* x, float* grad, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the thread is within the bounds of the input vector
    if (idx < N) {
        // Gradient of ReLU: 1 if x > 0, 0 if x <= 0
        grad[idx] = (x[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

void reluGradient(const float* x, float* grad, int N) {
    float* d_x, * d_grad;

    // Allocate memory on the device
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_grad, N * sizeof(float));

    // Copy the input data from host to device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block size for CUDA kernel
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize;  // Number of blocks needed

    // Launch the kernel
    reluGradientKernel << <numBlocks, blockSize >> > (d_x, d_grad, N);

    // Copy the result back from device to host
    cudaMemcpy(grad, d_grad, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the allocated memory on the device
    cudaFree(d_x);
    cudaFree(d_grad);
}

__global__ void rowMatrixMulKernel(float* row, float* matrix, float* result, int n, int m) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index of the result vector

    // Check if the thread is within the valid range of result vector's size
    if (col < m) {
        float sum = 0.0f;

        // Compute the dot product of row and the col-th column of matrix
        for (int i = 0; i < n; i++) {
            sum += row[i] * matrix[i * m + col]; // matrix[i * m + col] accesses element at row i, column col
        }

        // Store the result in the corresponding position
        result[col] = sum;
    }
}

#include <iostream>
#include <cuda_runtime.h>

void rowMatrixMul(float* row, float* matrix, float* result, int n, int m) {
    float* d_row, * d_matrix, * d_result;

    // Allocate memory on the device
    cudaMalloc(&d_row, n * sizeof(float));
    cudaMalloc(&d_matrix, n * m * sizeof(float));
    cudaMalloc(&d_result, m * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_row, row, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix, matrix, n * m * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (m + blockSize - 1) / blockSize;  // Ensure we cover all columns
    rowMatrixMulKernel << <numBlocks, blockSize >> > (d_row, d_matrix, d_result, n, m);

    // Copy result back to host
    cudaMemcpy(result, d_result, m * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_row);
    cudaFree(d_matrix);
    cudaFree(d_result);
}

__global__ void transposeKernel(float* input, float* output, int width, int height) {
    // Calculate row and column index in the transposed output matrix
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Perform transpose only if within matrix bounds
    if (x < width && y < height) {
        // Write transposed value from input to output
        output[x * height + y] = input[y * width + x];
    }
}

void transposeMatrix(float* d_input, float* d_output, int width, int height) {
    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    transposeKernel << <gridSize, blockSize >> > (d_input, d_output, width, height);
    cudaDeviceSynchronize();
}
void transposeMatrixCPU(float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

__global__ void rsmeKernel(float* pred, float* actual, float* output, int N) {
    extern __shared__ float temp[]; // Use shared memory to hold partial sums
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread computes squared difference for its data point, if within bounds
    temp[tid] = (idx < N) ? (pred[idx] - actual[idx]) * (pred[idx] - actual[idx]) : 0.0f;

    __syncthreads(); // Synchronize threads to prepare for reduction

    // Parallel reduction to sum all squared differences within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    // First thread of each block writes result to output
    if (tid == 0) {
        atomicAdd(output, temp[0]); // Atomic add across blocks
    }
}

// Host function to compute RMSE
float computeRMSE(float* pred, float* actual, int N) {
    float* d_pred, * d_actual, * d_output;
    float output = 0.0f;
    float* h_output = &output;

    cudaMalloc((void**)&d_pred, N * sizeof(float));
    cudaMalloc((void**)&d_actual, N * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));
    cudaMemcpy(d_pred, pred, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_actual, actual, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with one block and threads, each block handling part of the array
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    rsmeKernel << <numBlocks, blockSize, blockSize * sizeof(float) >> > (d_pred, d_actual, d_output, N);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate RMSE

    output = sqrtf(output / N);

    // Free memory
    cudaFree(d_pred);
    cudaFree(d_actual);
    cudaFree(d_output);

    return output;
}

__global__ void rsmeDerivativeKernel(float* pred, float* actual, float* grad, int N, float RMSE) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        grad[idx] = (pred[idx] - actual[idx]) / (N*RMSE);
    }
}

// Host function to compute RMSE derivative
void computeRMSEDerivative(float* pred, float* actual, float* grad, int N, float RMSE) {
    float* d_pred, * d_actual, * d_grad;

    cudaMalloc((void**)&d_pred, N * sizeof(float));
    cudaMalloc((void**)&d_actual, N * sizeof(float));
    cudaMalloc((void**)&d_grad, N * sizeof(float));
    cudaMemcpy(d_pred, pred, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_actual, actual, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    rsmeDerivativeKernel << <numBlocks, blockSize >> > (d_pred, d_actual, d_grad, N, RMSE);

    cudaMemcpy(grad, d_grad, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_pred);
    cudaFree(d_actual);
    cudaFree(d_grad);
}
__global__ void elementwiseMultiplyKernel(float* x, float* y, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        result[idx] = x[idx] * y[idx];
    }
}

void elementwiseMultiply(float* x, float* y, float* result, int N) {
    float* d_x, * d_y, * d_result;

    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMalloc((void**)&d_result, N * sizeof(float));

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    elementwiseMultiplyKernel << <numBlocks, blockSize >> > (d_x, d_y, d_result, N);

    cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
}