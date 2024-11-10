#pragma once
#include "kernel.cuh"
#include<random>
#include<algorithm>
#include<iostream>

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

