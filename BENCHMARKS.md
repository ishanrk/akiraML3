# Benchmarks

Benchmark suite runs MLP regression and classification on synthetic and UCI datasets. Target runtime under 1 hour.

## Run

Build with Configuration = Benchmark, Platform = x64. Run akiraML3.exe. Output CSV to stdout and benchmark_results.csv.

```bash
msbuild akiraML3.vcxproj /p:Configuration=Benchmark /p:Platform=x64
.\x64\Benchmark\akiraML3.exe
```

## Datasets

synthetic_reg_500, synthetic_reg_1000, synthetic_reg_2000: regression, 10 features, 30 epochs.  
synthetic_reg_5000: regression, 10 features, 20 epochs.  
synthetic_clf_500, synthetic_clf_1000: classification, 5 features, 3 classes, 30 epochs.  
iris: UCI Iris, 150 samples, 4 features, 3 classes, 50 epochs.  
wine: UCI Wine, 178 samples, 13 features, 3 classes, 50 epochs.

## Metrics

dataset, model, samples, features, epochs, train_sec, epoch_ms, samples_per_sec, final_loss, accuracy

train_sec: total training time in seconds.  
epoch_ms: milliseconds per epoch.  
samples_per_sec: samples * epochs / train_sec (throughput).  
final_loss: RMSE for regression.  
accuracy: fraction correct for classification (0 to 1).

## Dataloader

`dataloader.cuh` / `dataloader.cu`:

```cpp
#include "dataloader.cuh"

std::vector<std::vector<float>> data = loadCSV("data.csv");
std::vector<std::vector<float>> X, y;
splitDataRegression(data, num_features, X, y);

std::vector<int> labels;
splitDataClassification(data, num_features, num_classes, X, labels);
normalizeFeaturesMinMax(X);

auto [X_iris, labels_iris] = loadBuiltinIris();
auto [X_wine, labels_wine] = loadBuiltinWine();
```

CSV format: comma-separated, last column target (regression) or class index 0..k-1 (classification).

## Comparison with other C++ autodiff engines

To compare AkiraML with TinyDNN, mlpack, or dlib:

1. Run AkiraML benchmark, save benchmark_results.csv.
2. Implement same datasets and model configs in the other engine.
3. Match metrics: train_sec, epoch_ms, samples_per_sec, final_loss, accuracy.
4. Report results in same CSV format.

TinyDNN: header-only, C++14, https://github.com/tiny-dnn/tiny-dnn  
mlpack: C++ ML library, https://mlpack.org  
dlib: C++ toolkit with ML, http://dlib.net

Example comparison format:

| dataset | engine | train_sec | epoch_ms | samples_per_sec | final_loss | accuracy |
|---------|--------|-----------|----------|-----------------|------------|----------|
| iris | akiraML | X.XX | XX.X | XXXX | X.XXX | 0.XX |
| iris | TinyDNN | X.XX | XX.X | XXXX | X.XXX | 0.XX |

Run on same machine, same compiler (MSVC/Clang), Release build for fair comparison.
