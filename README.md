# AkiraML


CUDA C++ library for building and training ML models. Autodiff, variables, optimizers, MLP, transformer encoder.

Dependencies: CUDA, C++ standard library.

## What is Autodifferentiation
<img width="326" height="188" alt="image" src="https://github.com/user-attachments/assets/3c2df279-7596-4c7e-b5ba-a7009445249e" />

(Source: https://en.wikipedia.org/wiki/Automatic_differentiation#/media/File:ForwardAccumulationAutomaticDifferentiation.png)

As the name implies, this is simply an automated program to calculate derivatives of compositions of functions (which neural networks essentially are). This is done by analyzing the computational graph (think about the connections of a neural network) to calculate gradients at each node. Most of your favourite ML libraries like PyTorch have an underlying autodiff library written in C++ for speed reasons (quicker memory access, gpu kernel tricks). When you call .backward() in PyTorch it is the equivalent of effectively evaluating the derivatives we calculate in the computational graph based on your loss function (each derivative is of the form dL/d weight)

## Files

`variable.cuh` / `variable.cu` — tensor class and ops.  
`kernel.cuh` / `kernel.cu` — CUDA kernels.  
`optimizers.cuh` — Adam, RMSprop, SGD.  
`neural_network.cuh` / `neural_network.cu` — MLP.  
`models.cuh` / `models.cu` — linear regression.  
`transformers.cuh` / `transformers.cu` — transformer encoder.  
`dataloader.cuh` / `dataloader.cu` — load CSV, builtin Iris/Wine, normalize.  
`benchmark.cu` — benchmark suite (training speed / accuracy).

## Dataloader

Load CSV, split features and target, normalize.

```cpp
#include "dataloader.cuh"

std::vector<std::vector<float>> data = loadCSV("data.csv");
std::vector<std::vector<float>> X, y;
splitDataRegression(data, num_features, X, y);

std::vector<int> labels;
splitDataClassification(data, num_features, num_classes, X, labels);
normalizeFeaturesMinMax(X);
```

Builtin datasets: Iris (150, 4, 3 classes), Wine (178, 13, 3 classes).

```cpp
auto [X, labels] = loadBuiltinIris();
auto [X2, labels2] = loadBuiltinWine();
normalizeFeaturesMinMax(X);
```

## MNIST / Fashion-MNIST Example

1. **Download data (Python, one-time):**  
   `pip install torch torchvision`  
   `python scripts/download_mnist.py`  
   Creates `data/mnist_train.csv`, `data/mnist_test.csv`, `data/fashion_mnist_*.csv` (785 cols: 784 pixels + label).

2. **Build and train (C++):**  
   `nvcc -std=c++17 -O2 -o build/train_mnist train_mnist.cu neural_network.cu variable.cu kernel.cu dataloader.cu`  
   `build/train_mnist` — MNIST, 10 epochs.  
   `build/train_mnist --fashion` — Fashion-MNIST.  
   `build/train_mnist --epochs 20 --save data/my_mlp.akira`

The MLP is 784 → 256 → 128 → 10 (ReLU, ReLU, Softmax) with Adam.

## Tests and validation

Fast functional tests and a verbose validation suite live in `tests.cu` / `test.cu`.

- **Non–NN tests only** (variables, kernels, optimizers, linear regression):
  - `nvcc -std=c++17 -O2 -o build/run_tests test.cu tests.cu variable.cu kernel.cu models.cu dataloader.cu`
  - `build/run_tests`
- **Full suite + validation (diverse regression / classification with lots of logs)**:
  - same command as above; `test.cu` runs:
    - `runNonNeuralNetworkTests()` (low‑level ops)
    - neural‑network regression / classification unit tests
    - `runValidationSuite()` (small/medium/edge datasets, verbose epoch‑by‑epoch loss and timings)

## Benchmark

`benchmark.cu` runs **synthetic regression / classification** and **UCI Iris/Wine** benchmarks and writes a CSV line per run.

- **Build and run:**
  - `nvcc -std=c++17 -O2 -o build/benchmark benchmark.cu neural_network.cu variable.cu kernel.cu models.cu dataloader.cu`
  - `build/benchmark`
- **Output:**
  - Prints a CSV header and rows to stdout
  - Writes `benchmark_results.csv` with columns  
    `engine,dataset,model,samples,features,epochs,train_sec,epoch_ms,samples_per_sec,final_loss,accuracy`  
    where `engine` is `akiraML3` by default.
- **Graphs (time, speed, accuracy):**
  - `pip install -r scripts/requirements.txt`
  - `python scripts/plot_benchmarks.py benchmark_results.csv -o benchmarks/plots`
  - See `benchmarks/README.md` for how to add results from other C++ autodiff engines
    (e.g. LibTorch, `autodiff`, Eigen‑based engines) using the same CSV schema.

## Variable

Create a variable: (rows, cols, random). Print as matrix or vector.

```cpp
#include "variable.cuh"

variable var(2, 2, true);
var.print(true);
```

Matrix-vector multiply.

```cpp
variable mat(3, 3, true);
variable vec(3, 1, true);
variable result = mat.matrixMulVec(vec);
result.print();
```

Matrix-matrix multiply.

```cpp
variable A(4, 3, true);
variable B(3, 5, true);
variable C = A.matrixMul(B);
C.print(true);
```

Transpose, scale, element-wise add.

```cpp
variable x(2, 3, false);
x.setData(some_data);
variable xt = x.transpose();
variable scaled = x.scale(0.5f);
variable y(2, 3, false);
y.setData(other_data);
variable z = x + y;
```

Activations: relu, sigmoid, softmax, rowSoftmax.

```cpp
variable in(4, 1, true);
variable a = in.relu();
variable b = in.sigmoid();
variable c = in.softmax();

variable mat(3, 4, true);
variable row_soft = mat.rowSoftmax();
```

RMSE loss and backward.

```cpp
variable pred(5, 1, false);
variable target(5, 1, false);
pred.setData(pred_data);
target.setData(target_data);
variable loss = pred.RMSELOSS(target);
float* arr = nullptr;
loss.backward(&loss, arr, 0);
```

## Linear regression

SGD.

```cpp
#include "models.cuh"

std::vector<std::pair<float, float>> dataset = generateLinearData(100, 2.5f, 1.0f, 0.1f);
auto result = scalarLinearRegression(dataset, 0.001f);
std::cout << result.first << " " << result.second << std::endl;
```

Adam.

```cpp
#include "models.cuh"
#include "optimizers.cuh"

auto dataset = generateLinearData(100, 2.5f, 1.0f, 0.1f);
auto result = scalarLinearRegressionAdam(dataset, 0.01f);
std::cout << result.first << " " << result.second << std::endl;
```

RMSprop.

```cpp
#include "models.cuh"
#include "optimizers.cuh"

auto dataset = generateLinearData(100, 2.5f, 1.0f, 0.1f);
auto result = scalarLinearRegressionRMSprop(dataset, 0.01f);
std::cout << result.first << " " << result.second << std::endl;
```

## Neural network regression

```cpp
#include "neural_network.cuh"
#include "optimizers.cuh"

auto data = generateRegressionData(200, 2);
std::vector<std::vector<float>> X, y;
for (const auto& s : data) {
    X.push_back({s[0], s[1]});
    y.push_back({s[2]});
}

std::vector<int> sizes = {2, 8, 4, 1};
std::vector<std::string> acts = {"relu", "relu", "linear"};
auto opt = std::make_shared<Adam>(0.01f, 0.9f, 0.999f, 1e-8f);
NeuralNetwork nn(sizes, acts, opt);
nn.printArchitecture();
nn.train(X, y, 100, 0.01f);

std::vector<float> test = {1.0f, 0.5f};
auto out = nn.predict(test);
std::cout << out[0] << std::endl;
```

## Neural network classification

```cpp
#include "neural_network.cuh"
#include "optimizers.cuh"
#include <algorithm>

auto data = generateClassificationData(150, 2, 3);
std::vector<std::vector<float>> X;
std::vector<int> labels;
for (const auto& s : data) {
    X.push_back({s[0], s[1]});
    labels.push_back(static_cast<int>(s[2]));
}
auto y = oneHotEncode(labels, 3);

std::vector<int> sizes = {2, 6, 3};
std::vector<std::string> acts = {"relu", "softmax"};
auto opt = std::make_shared<RMSprop>(0.01f, 0.9f, 1e-8f);
NeuralNetwork nn(sizes, acts, opt);
nn.train(X, y, 200, 0.01f);

std::vector<float> test = {1.0f, 0.5f};
auto pred = nn.predict(test);
int cls = std::max_element(pred.begin(), pred.end()) - pred.begin();
std::cout << cls << std::endl;
```

## Transformer encoder

```cpp
#include "transformers.cuh"
#include "optimizers.cuh"

int seq_len = 8;
int d_model = 64;
int num_heads = 4;
int d_ff = 256;
int num_layers = 2;

auto opt = std::make_shared<Adam>(0.0001f, 0.9f, 0.999f, 1e-8f);
TransformerEncoder encoder(seq_len, d_model, num_heads, d_ff, num_layers, opt);

variable input(seq_len, d_model, false);
// set input.data to your embeddings

variable output = encoder.forward(input);
```

## Save and load

Models use the `.akira` text format. One number per line for floats and ints. Keywords on their own line.

Save and load MLP.

```cpp
#include "neural_network.cuh"

NeuralNetwork nn(sizes, acts, opt);
nn.train(X, y, 100, 0.01f);
nn.save("model.akira");

NeuralNetwork loaded = NeuralNetwork::load("model.akira");
std::vector<float> out = loaded.predict(test);
```

Save and load transformer.

```cpp
#include "transformers.cuh"

TransformerEncoder enc(8, 64, 4, 256, 2, opt);
enc.save("transformer.akira");

TransformerEncoder loaded = TransformerEncoder::load("transformer.akira");
variable out = loaded.forward(input);
```

## .akira format

File is UTF-8 text. Read line by line or token by token. First line is model type.

MLP file layout:

```
MLP
LAYERS
<num_layers>
<layer_sizes[0]>
<layer_sizes[1]>
...
ACTIVATIONS
<num_activations>
<act0>
<act1>
...
WEIGHT
<layer_index>
<rows> <cols>
<float> (rows*cols lines)
WEIGHT
...
BIAS
<layer_index>
<dim>
<float> (dim lines)
...
END
```

Example: 2 inputs, 4 hidden, 1 output, activations relu and linear.

```
MLP
LAYERS
3
2
4
1
ACTIVATIONS
2
relu
linear
WEIGHT
0
4 2
<float x8>
WEIGHT
1
1 4
<float x4>
BIAS
0
4
<float x4>
BIAS
1
1
<float x1>
END
```

Transformer file layout:

```
TRANSFORMER
MAX_LEN
<max_len>
D_MODEL
<d_model>
NUM_HEADS
<num_heads>
D_FF
<d_ff>
NUM_LAYERS
<num_layers>
LAYER
<layer_index>
WQ
<rows> <cols>
<float> (rows*cols lines)
WK
...
WV
...
WO
...
WFF1
...
BFF1
...
WFF2
...
BFF2
...
LAYER
...
PE
<max_len> <d_model>
<float> (max_len*d_model lines)
END
```

Weights are row-major. WQ, WK, WV, WO are (d_model, d_model). WFF1 (d_model, d_ff). BFF1 (seq_len, d_ff). WFF2 (d_ff, d_model). BFF2 (seq_len, d_model).

Define a model in code, train or set weights, then save to a path. Load from path returns a new model with the same layout and filled weights. Optimizer is not stored; set it after load if you will train.

```cpp
NeuralNetwork nn = NeuralNetwork::load("model.akira");
nn.setOptimizer(std::make_shared<Adam>(0.001f, 0.9f, 0.999f, 1e-8f));
nn.train(X, y, 10, 0.001f);
nn.save("model.akira");
```

Scaled dot-product attention only (Q, K, V variables).

```cpp
#include "transformers.cuh"

variable Q(seq_len, d_k, false);
variable K(seq_len, d_k, false);
variable V(seq_len, d_v, false);
// set Q.data, K.data, V.data
variable attn_out = scaledDotProductAttention(Q, K, V);
```

Positional encoding buffer (no learnable params).

```cpp
float* pe = (float*)malloc(max_len * d_model * sizeof(float));
sinusoidalPositionalEncoding(pe, max_len, d_model);
// add pe to input embeddings, then free(pe)
```

## Backward and optimizers

Run backward from loss, then update parameters with optimizer.

```cpp
variable loss = model_output.RMSELOSS(target);
float* arr = nullptr;
loss.backward(&loss, arr, 0);

for (auto& w : weights) {
    w.updateWithOptimizer(iteration);
}
```

Set optimizer on a variable.

```cpp
auto adam = std::make_shared<Adam>(0.001f, 0.9f, 0.999f, 1e-8f);
weight.setOptimizer(adam);
```

## Computational graph

Operations build a graph. Backward propagates gradients from root to leaves.

```cpp
variable x(2, 1, false);
variable y(2, 1, false);
x.setData(x_data);
y.setData(y_data);
variable z = x + y;
variable out = z.relu();
variable loss = out.RMSELOSS(target);
loss.backward(&loss, nullptr, 0);
// x.backwardGrad, y.backwardGrad filled
```

## Kernel usage

Matrix-vector multiply on device.

```cpp
void matrixVectorMul(float* A, float* x, float* y, int M, int N);
```

Matrix-matrix multiply.

```cpp
void matrixMatrixMul(float* A, float* B, float* C, int M, int K, int N);
```

Row-wise softmax (e.g. for attention weights).

```cpp
void rowSoftmax(float* input, float* output, int rows, int cols);
```

Transpose.

```cpp
void transposeMatrixCPU(float* input, float* output, int rows, int cols);
```

Element-wise multiply.

```cpp
void elementwiseMultiply(float* x, float* y, float* result, int N);
```

## Adding a custom op

Implement forward, store gradients for each input in result.gradientChild1 (and gradientChild2 for binary ops), push parents so backward can traverse.

```cpp
variable myOp(variable& a) {
    std::vector<variable*> temp = {&a};
    variable result(a.dim1, a.dim2, false, temp);
    result.data = (float*)malloc(a.dim1 * a.dim2 * sizeof(float));
    result.gradientChild1 = (float*)malloc(a.dim1 * a.dim2 * sizeof(float));
    for (int i = 0; i < a.dim1 * a.dim2; i++) {
        result.data[i] = a.data[i] * a.data[i];
        result.gradientChild1[i] = 2.0f * a.data[i];
    }
    a.parents.push_back(result);
    return result;
}
```

Add a branch in variable::backward that detects this op (e.g. by dimensions or a sentinel) and computes backwardGrad from gradAccum and the stored gradients.

## Performance

The benchmark suite covers:

1. **Regression examples:** synthetic `reg_n{100,250,500,1000,2000,5000}` with 10 features and MLP `[10,16,8,1]`
2. **Classification examples:** synthetic `clf_n{200,500,1000,2000}` with 5 features, 3 classes and MLP `[5,8,3]`
3. **UCI datasets:** Iris (`[4,8,3]`) and Wine (`[13,16,3]`)

For each run it reports: **`train_sec`**, **`epoch_ms`**, **`samples_per_sec`**, **`final_loss`**, and (for classification) **`accuracy`**.  
Use `scripts/plot_benchmarks.py` to generate comparison graphs for `akiraML3` and any other C++ autodiff engines that emit the same CSV format (see `benchmarks/README.md`).
