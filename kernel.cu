#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
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
    }
}
// Helper function for using CUDA to add vectors in parallel.
int addWithCuda(float* c, float* a, float* b, int dim1)
{
    int id = cudaGetDevice(&id);
    int BLOCKSZ = 4;
    int GRIDSZ = (int)ceil(dim1 / BLOCKSZ);

    vectorAddUM <<<GRIDSZ, BLOCKSZ >>> (c, a, b, dim1);

    cudaDeviceSynchronize();

    return 1;

}

