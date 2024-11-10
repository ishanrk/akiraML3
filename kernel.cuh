#pragma once
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
// Function declaration
void random_init(float* data, int dim1);
__global__ void vectorAddUM(float* c, float* a, float* b, int dim1);
int addWithCuda(float* c, float* a, float* b, int dim1);