# AkiraML Library

AkiraML is a CUDA-based C++ library designed for building and training machine learning models with support for custom variables, backpropagation, activation functions, and basic operations like matrix multiplication and element-wise operations. This library is an early prototype and currently supports core features needed for defining and manipulating tensors (variables) and performing basic operations.

## Features
- **Variable Class**: Supports tensor-like variable handling, which includes initialization, addition, matrix-vector multiplication, element-wise operations, and scaling.
- **Backpropagation**: Allows for basic gradient calculation with respect to operations.
- **Activation Functions**: Includes ReLU, Sigmoid, and Softmax functions with CUDA-based forward and backward (gradient) computations.
- **RMSE Loss**: Supports calculation of Root Mean Square Error loss for training purposes.
- **Linear Regression Model**: An example machine learning model is included to demonstrate how to use AkiraML for training with synthetic data.

## Dependencies
1. **CUDA**: Required for GPU computations.
2. **C++ Standard Libraries**: For vector and I/O operations.

## Files and Structure
- **`variable.h`**: Contains the `variable` class, defining the core data structure for storing and manipulating tensors.
- **`kernel.cuh`**: Defines CUDA kernels and CUDA-based utility functions for matrix operations, activation functions, etc.
- **`models.cuh`**: Contains basic machine learning models, such as a scalar linear regression example.
- **`main.cpp`**: Demonstrates the usage of AkiraML with a simple scalar linear regression model.

## Getting Started

### Step 1: Initialize a Variable
```cpp
#include "variable.h"

int main() {
    // Create a variable with dimensions (2, 2) and random initialization
    variable var(2, 2, true);
    var.print(true); // Print the variable as a matrix
    return 0;
}
```

### Step 2: Perform Matrix-Vector Multiplication
```cpp

variable mat(3, 3, true);   // Random 3x3 matrix
variable vec(3, 1, true);   // Random 3x1 vector

// Perform matrix-vector multiplication
variable result = mat.matrixMulVec(vec);
result.print();
```
### Step 3: Use Activation Functions

```
variable input(3, 1, true);

// Apply ReLU activation
variable relu_output = input.relu();
relu_output.print();

// Apply Sigmoid activation
variable sigmoid_output = input.sigmoid();
sigmoid_output.print();

```
### Step 4: Calculate Loss with RMSE
```
variable predicted(3, 1, true);
variable actual(3, 1, true);

// Calculate RMSE loss
variable loss = predicted.RMSELOSS(actual);
loss.print();
```
### Step 5: Run a Simple Linear Regression Model
A linear regression example with CUDA-based optimizations is provided in the main.cpp file.
```
#include <iostream>
#include "models.cuh"

int main() {
    int num_samples = 10;
    float slope = 10000;      // Specify slope (m)
    float intercept = -988;   // Specify intercept (c)
    float noise_stddev = 0.00001; // Standard deviation of noise

    // Generate the dataset
    std::vector<std::pair<float, float>> dataset = generateLinearData(num_samples, slope, intercept, noise_stddev);

    // Perform scalar linear regression
    auto result = scalarLinearRegression(dataset, 0.001); // Learning rate = 0.001
    std::cout << "Predicted slope: " << result.first << ", intercept: " << result.second << std::endl;

    return 0;
}
```
## Library Components
### Class: variable
Represents a tensor with dimensions (dim1, dim2) and supports operations like:

- Addition: operator+
- Matrix-Vector Multiplication: matrixMulVec
- Activation Functions: relu, sigmoid, softmax
- Backpropagation: backward through constructed computational graph
- RMSE Loss: RMSELOSS
- Scaling: scale
  
### CUDA Kernels (kernel.cuh)
- Defines GPU kernels for various operations:
- vectorAddUM: Performs element-wise addition.
- multiplyVectorsKernel: Multiplies vectors element-wise.
- matrixVectorMulKernel: Multiplies a matrix with a vector.
- sigmoidKernel: Applies the sigmoid function.
- reluKernel: Applies the ReLU function.
- softmaxKernel: Applies the softmax function.
  
## Example CUDA Kernel Usage
The library uses CUDA kernels to perform operations in parallel. For example, the matrixVectorMulKernel kernel multiplies a matrix with a vector:
```
__global__ void matrixVectorMulKernel(float* A, float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float dot_product = 0;
        for (int col = 0; col < N; ++col) {
            dot_product += A[row * N + col] * x[col];
        }
        y[row] = dot_product;
    }
}
```
## Future Improvements
This library is currently a work in progress. Some potential future improvements:

- Define thresholds for GPU operations to switch over from CPU operations (1024 element size)
- Implement additional machine learning models (e.g., logistic regression, neural networks).
- Add support for more advanced optimizers (e.g., Adam, RMSprop).
- Include more sophisticated backpropagation mechanisms for deeper models.
- Improve documentation and provide a larger set of examples for end-users.
