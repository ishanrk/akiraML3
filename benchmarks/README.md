# Benchmarking akiraML3 vs other C++ autodiff engines

This folder holds benchmark results and plots comparing **akiraML3** (CUDA-backed autodiff MLP) with other C++ autodiff engines on the same tasks.

## Quick start

1. **Run the C++ benchmark** (from project root, in a Developer Command Prompt with CUDA):

   ```bat
   nvcc -std=c++17 -O2 -o build\benchmark.exe benchmark.cu neural_network.cu variable.cu kernel.cu models.cu dataloader.cu
   build\benchmark.exe
   ```

   This writes `benchmark_results.csv` with columns:  
   `engine,dataset,model,samples,features,epochs,train_sec,epoch_ms,samples_per_sec,final_loss,accuracy`

2. **Generate graphs** (from project root):

   ```bat
   pip install -r scripts/requirements.txt
   python scripts/plot_benchmarks.py
   ```

   Plots are saved under `benchmarks/plots/`:
   - `train_sec_vs_samples_regression.png` – training time vs dataset size (regression)
   - `train_sec_vs_samples_classification.png` – same for classification
   - `epoch_ms_by_dataset.png` – time per epoch by dataset
   - `samples_per_sec_by_dataset.png` – throughput by dataset
   - `train_sec_by_dataset.png` – total training time by dataset
   - `final_loss_by_dataset.png` – final loss
   - `accuracy_by_dataset.png` – classification accuracy

## Comparing with other C++ autodiff engines

To compare akiraML3 with **LibTorch**, **autodiff**, **Eigen AutoDiff**, or similar:

1. **Same CSV schema**  
   Produce a CSV with the same column names and types:
   - `engine` – e.g. `LibTorch`, `autodiff`, `akiraML3`
   - `dataset` – same dataset names (e.g. `reg_n500`, `synthetic_clf_500`, `iris`)
   - `model` – string description of the model
   - `samples`, `features`, `epochs` – integers
   - `train_sec`, `epoch_ms`, `samples_per_sec` – floats
   - `final_loss`, `accuracy` – float (accuracy 0–1 for classification)

2. **Same tasks**  
   For a fair comparison, run the same workloads:
   - **Regression sweep:** samples = 100, 250, 500, 1000, 2000, 5000; 10 features; MLP [10,16,8,1]; 30 epochs.
   - **Classification sweep:** samples = 200, 500, 1000, 2000; 5 features, 3 classes; MLP [5,8,3]; 30 epochs.
   - **Iris:** 150 samples, 4 features, 3 classes; MLP [4,8,3]; 100 epochs.
   - **Wine:** 178 samples, 13 features, 3 classes; MLP [13,16,3]; 50 epochs.

   Use the same data generation (or load the same CSV data) so only the engine differs.

3. **Merge and plot**  
   Pass multiple CSVs to the plotting script:

   ```bat
   python scripts/plot_benchmarks.py benchmark_results.csv libtorch_results.csv autodiff_results.csv -o benchmarks/plots
   ```

   The script merges all rows and plots by `engine`, so you get side-by-side bars or multiple lines for speed and time.

## Reference: C++ engines to consider

| Engine       | Notes                          |
|-------------|---------------------------------|
| **LibTorch** | PyTorch C++ API; full autograd. |
| **autodiff** | Header-only C++17; forward/reverse. |
| **Eigen**    | Unsupported AutoDiff module.   |
| **XAD**      | C++ AD with expression templates. |

You can add placeholder or real result CSVs for any of these using the same schema and dataset names above.
