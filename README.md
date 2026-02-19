# AkiraML Library

AkiraML is a CUDA-based C++ library designed for building and training machine learning models with support for custom variables, backpropagation, activation functions, and basic operations like matrix multiplication and element-wise operations. This library is an early prototype and currently supports core features needed for defining and manipulating tensors (variables) and performing basic operations.

## Features
- **Auto-Differentiation Engine**: Implements reverse-mode automatic differentiation with computational graph construction, chain rule gradient computation, and Jacobian matrix calculations for vector-valued functions.
- **Neural Network Framework**: Complete multi-layer perceptron implementation supporting both regression and classification tasks with configurable architectures, batch processing, and one-hot encoding utilities.
- **Advanced Optimizers**: Three CUDA-accelerated optimization algorithms (Adam, RMSprop, SGD) with adaptive learning rates, momentum, and second-order moment estimation for efficient parameter updates.
- **CUDA-Accelerated Operations**: 15+ parallel GPU kernels for matrix-vector multiplication, element-wise operations, activation functions (ReLU, Sigmoid, Softmax), gradient computations, and matrix transposition.
- **Tensor Operations**: Comprehensive variable class supporting addition, dot product, matrix-vector and matrix-matrix multiplication, scaling, transpose, row-wise softmax, and seamless integration with computational graphs for automatic backpropagation.
- **Transformers**: Transformer encoder with scaled dot-product attention, sinusoidal positional encoding, multi-head self-attention (configurable heads), and feed-forward layers with residual connections; all differentiable and CUDA-backed.
- **Comprehensive Testing**: 13+ test cases covering vector operations, activation functions, loss calculations, linear regression models, neural network training, and optimizer performance comparisons with synthetic data generation.

## Dependencies
1. **CUDA**: Required for GPU computations.
2. **C++ Standard Libraries**: For vector and I/O operations.

## Files and Structure
- **`variable.h`**: Contains the `variable` class, defining the core data structure for storing and manipulating tensors.
- **`kernel.cuh`**: Defines CUDA kernels and CUDA-based utility functions for matrix operations, activation functions, etc.
- **`models.cuh`** / **`models.cu`**: Basic machine learning models (e.g. scalar linear regression with SGD, Adam, RMSprop).
- **`transformers.cuh`** / **`transformers.cu`**: Transformer encoder: scaled dot-product attention, positional encoding, `TransformerEncoderLayer`, and `TransformerEncoder`.

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
- Matrix-Matrix Multiplication: matrixMul
- Transpose: transpose
- Activation Functions: relu, sigmoid, softmax, rowSoftmax (per-row softmax for attention)
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
## Advanced Examples

### RMSprop Optimizer Example
```cpp
#include <iostream>
#include "models.cuh"
#include "optimizers.cuh"

int main() {
    // Generate synthetic linear data
    int num_samples = 100;
    float slope = 2.5f;
    float intercept = 1.0f;
    float noise_stddev = 0.1f;
    
    std::vector<std::pair<float, float>> dataset = generateLinearData(num_samples, slope, intercept, noise_stddev);
    
    // Train using RMSprop optimizer
    std::pair<float, float> result = scalarLinearRegressionRMSprop(dataset, 0.01f);
    
    std::cout << "True slope: " << slope << ", intercept: " << intercept << std::endl;
    std::cout << "Predicted slope: " << result.first << ", intercept: " << result.second << std::endl;
    
    return 0;
}
```

### Neural Network Example
```cpp
#include <iostream>
#include "neural_network.cuh"
#include "optimizers.cuh"

int main() {
    // Generate regression data
    auto data = generateRegressionData(200, 2);  // 200 samples, 2 features
    
    // Separate features and targets
    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> y;
    
    for (const auto& sample : data) {
        X.push_back({sample[0], sample[1]});  // Features
        y.push_back({sample[2]});             // Target
    }
    
    // Create neural network: 2 inputs -> 8 hidden -> 4 hidden -> 1 output
    std::vector<int> layer_sizes = {2, 8, 4, 1};
    std::vector<std::string> activations = {"relu", "relu", "linear"};
    
    // Use Adam optimizer
    auto adam_opt = std::make_shared<Adam>(0.01f, 0.9f, 0.999f, 1e-8f);
    NeuralNetwork nn(layer_sizes, activations, adam_opt);
    
    // Print network architecture
    nn.printArchitecture();
    
    // Train the network
    nn.train(X, y, 100, 0.01f);
    
    // Make predictions
    std::vector<float> test_input = {1.0f, 0.5f};
    auto prediction = nn.predict(test_input);
    
    std::cout << "Test input: [" << test_input[0] << ", " << test_input[1] << "]" << std::endl;
    std::cout << "Prediction: " << prediction[0] << std::endl;
    
    return 0;
}
```

### Transformer Encoder Example
```cpp
#include <iostream>
#include "transformers.cuh"
#include "optimizers.cuh"

int main() {
    int seq_len = 8;   // Sequence length
    int d_model = 64;  // Model dimension
    int num_heads = 4;
    int d_ff = 256;    // Feed-forward hidden dimension
    int num_layers = 2;

    auto opt = std::make_shared<Adam>(0.0001f, 0.9f, 0.999f, 1e-8f);
    TransformerEncoder encoder(seq_len, d_model, num_heads, d_ff, num_layers, opt);

    variable input(seq_len, d_model, false);
    // ... set input.data for your sequence embeddings ...

    variable output = encoder.forward(input);
    // output has shape (seq_len, d_model); use with a loss and backward() for training
    return 0;
}
```

### Classification Neural Network Example
```cpp
#include <iostream>
#include "neural_network.cuh"
#include "optimizers.cuh"

int main() {
    // Generate classification data
    auto data = generateClassificationData(150, 2, 3);  // 150 samples, 2 features, 3 classes
    
    // Separate features and labels
    std::vector<std::vector<float>> X;
    std::vector<int> labels;
    
    for (const auto& sample : data) {
        X.push_back({sample[0], sample[1]});  // Features
        labels.push_back(static_cast<int>(sample[2]));  // Class labels
    }
    
    // One-hot encode labels
    auto y = oneHotEncode(labels, 3);
    
    // Create neural network: 2 inputs -> 6 hidden -> 3 outputs
    std::vector<int> layer_sizes = {2, 6, 3};
    std::vector<std::string> activations = {"relu", "softmax"};
    
    // Use RMSprop optimizer
    auto rmsprop_opt = std::make_shared<RMSprop>(0.01f, 0.9f, 1e-8f);
    NeuralNetwork nn(layer_sizes, activations, rmsprop_opt);
    
    // Train the network
    nn.train(X, y, 200, 0.01f);
    
    // Make predictions
    std::vector<float> test_input = {1.0f, 0.5f};
    auto prediction = nn.predict(test_input);
    
    std::cout << "Test input: [" << test_input[0] << ", " << test_input[1] << "]" << std::endl;
    std::cout << "Class probabilities: [" << prediction[0] << ", " << prediction[1] << ", " << prediction[2] << "]" << std::endl;
    
    // Find predicted class
    int predicted_class = std::max_element(prediction.begin(), prediction.end()) - prediction.begin();
    std::cout << "Predicted class: " << predicted_class << std::endl;
    
    return 0;
}
```

## Autodifferentiation Engine

### How AkiraML's Autodifferentiation Works

AkiraML implements a **reverse-mode automatic differentiation** engine that builds computational graphs during forward passes and computes gradients during backward passes. Here's how it works:

#### 1. Computational Graph Construction
Every operation in AkiraML creates a new `variable` object that maintains:
- **Data**: The computed values
- **Children**: References to input variables
- **Parents**: References to output variables  
- **Gradients**: Pre-computed gradient information for each child

```cpp
// Example: z = x + y creates a computational graph
variable x(2, 1, false);
variable y(2, 1, false);
variable z = x + y;  // Creates graph: x -> z <- y
```

#### 2. Gradient Storage
Each operation pre-computes and stores the gradients with respect to its inputs:

```cpp
// In addition operation (x + y):
// gradientChild1 = [1, 1, 1, ...] (gradient w.r.t. x)
// gradientChild2 = [1, 1, 1, ...] (gradient w.r.t. y)
```

#### 3. Backward Pass
The `backward()` method traverses the computational graph in reverse order:

```cpp
// Start from loss and propagate gradients backward
loss.backward(&loss, nullptr, 0);
```

The backward pass:
1. **Initializes** the root gradient to 1.0
2. **Traverses** the graph from children to parents
3. **Accumulates** gradients using the chain rule
4. **Stores** final gradients in `backwardGrad` for each variable

#### 4. Chain Rule Implementation
For each operation, gradients are computed using the chain rule:

```cpp
// If z = f(x, y) and we have ∂L/∂z, then:
// ∂L/∂x = ∂L/∂z * ∂z/∂x
// ∂L/∂y = ∂L/∂z * ∂z/∂y
```

### Jacobians in AkiraML

AkiraML computes **Jacobian matrices** for vector-valued functions, particularly for activation functions:

#### 1. Element-wise Operations
For element-wise operations like ReLU and Sigmoid, the Jacobian is diagonal:

```cpp
// ReLU: J[i,i] = 1 if x[i] > 0, else 0
// Sigmoid: J[i,i] = σ(x[i]) * (1 - σ(x[i]))
```

#### 2. Softmax Jacobian
Softmax computes the full Jacobian matrix since each output depends on all inputs:

```cpp
// Softmax Jacobian: J[i,j] = σ(x[i]) * (δ[i,j] - σ(x[j]))
// where δ[i,j] is the Kronecker delta
```

#### 3. Matrix Operations
For matrix-vector multiplication `y = Ax`, the Jacobian is:
- **w.r.t. A**: `x^T` (transposed input vector)
- **w.r.t. x**: `A^T` (transposed weight matrix)

### Key Features of AkiraML's Autodiff

1. **CUDA Acceleration**: All gradient computations run on GPU for performance
2. **Memory Efficient**: Gradients are computed on-demand during backward pass
3. **Extensible**: New operations can be added by implementing forward/backward functions
4. **Optimizer Integration**: Gradients are automatically used by optimizers (Adam, RMSprop, SGD)

### Example: Custom Operation with Autodiff
```cpp
// To add a new operation, you need to:
// 1. Implement forward computation
// 2. Pre-compute gradients w.r.t. inputs
// 3. Store gradients in gradientChild1, gradientChild2, etc.

variable customOperation(variable& input) {
    // Forward pass
    variable result = /* compute result */;
    
    // Pre-compute gradients (example: square operation)
    // gradient = 2 * input (derivative of x² is 2x)
    for (int i = 0; i < input.dim1 * input.dim2; i++) {
        result.gradientChild1[i] = 2.0f * input.data[i];
    }
    
    return result;
}
```

