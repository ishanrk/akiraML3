#include "neural_network.cuh"
#include "optimizers.cuh"
#include "dataloader.cuh"
#include <iostream>
#include <tuple>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

struct BenchmarkResult {
    std::string engine;
    std::string dataset;
    std::string model;
    int samples;
    int features;
    int epochs;
    double train_sec;
    double epoch_ms;
    double samples_per_sec;
    float final_loss;
    float accuracy;
};

static float computeAccuracy(NeuralNetwork& nn,
    const std::vector<std::vector<float>>& X, const std::vector<int>& labels) {
    if (X.empty() || labels.empty()) return 0.0f;
    int correct = 0;
    for (size_t i = 0; i < X.size(); i++) {
        auto pred = nn.predict(X[i]);
        int predClass = static_cast<int>(std::max_element(pred.begin(), pred.end()) - pred.begin());
        if (predClass == labels[i]) correct++;
    }
    return static_cast<float>(correct) / static_cast<float>(labels.size());
}

static void runRegBenchmark(const std::string& engine, const std::string& name, int n, int feat, int epochs,
    std::vector<int> layer_sizes, std::vector<BenchmarkResult>& results) {
    auto data = generateRegressionData(n, feat);
    std::vector<std::vector<float>> X, y;
    splitDataRegression(data, feat, X, y);
    if (layer_sizes.empty()) layer_sizes = { feat, 16, 8, 1 };
    std::vector<std::string> activations(layer_sizes.size() - 1, "relu");
    activations.back() = "linear";
    auto opt = std::make_shared<Adam>(0.01f, 0.9f, 0.999f, 1e-8f);
    NeuralNetwork nn(layer_sizes, activations, opt);

    auto start = std::chrono::high_resolution_clock::now();
    nn.train(X, y, epochs, 0.01f, false);
    auto end = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(end - start).count();

    std::string model_str = "MLP[";
    for (size_t i = 0; i < layer_sizes.size(); i++) model_str += (i ? "," : "") + std::to_string(layer_sizes[i]);
    model_str += "]";
    BenchmarkResult r;
    r.engine = engine;
    r.dataset = name;
    r.model = model_str;
    r.samples = n;
    r.features = feat;
    r.epochs = epochs;
    r.train_sec = sec;
    r.epoch_ms = (epochs > 0) ? (sec * 1000.0 / epochs) : 0.0;
    r.samples_per_sec = (sec > 0) ? (n * epochs / sec) : 0.0;
    r.final_loss = nn.calculateLoss(X, y);
    r.accuracy = 0.0f;
    results.push_back(r);
}

static void runClfBenchmark(const std::string& engine, const std::string& name,
    const std::vector<std::vector<float>>& X, const std::vector<int>& labels,
    int feat, int numClasses, const std::vector<int>& arch, int epochs,
    std::vector<BenchmarkResult>& results) {
    auto y = oneHotEncode(labels, numClasses);
    auto opt = std::make_shared<Adam>(0.01f, 0.9f, 0.999f, 1e-8f);
    NeuralNetwork nn(arch, { "relu", "softmax" }, opt);

    auto start = std::chrono::high_resolution_clock::now();
    nn.train(X, y, epochs, 0.01f, false);
    auto end = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(end - start).count();

    BenchmarkResult r;
    r.engine = engine;
    r.dataset = name;
    r.model = "MLP";
    r.samples = static_cast<int>(X.size());
    r.features = feat;
    r.epochs = epochs;
    r.train_sec = sec;
    r.epoch_ms = (epochs > 0) ? (sec * 1000.0 / epochs) : 0.0;
    r.samples_per_sec = (sec > 0) ? (X.size() * epochs / sec) : 0.0;
    r.final_loss = nn.calculateLoss(X, y);
    r.accuracy = computeAccuracy(nn, X, labels);
    results.push_back(r);
}

static void writeCsvEscaped(std::ostream& out, const std::string& s) {
    if (s.find(',') != std::string::npos || s.find('"') != std::string::npos) {
        out << "\"";
        for (char c : s) { if (c == '"') out << "\"\""; else out << c; }
        out << "\"";
    } else out << s;
}

static void printCsvRow(const BenchmarkResult& r) {
    std::cout << r.engine << "," << r.dataset << ",";
    writeCsvEscaped(std::cout, r.model);
    std::cout << "," << r.samples << "," << r.features << "," << r.epochs << "," << r.train_sec << "," << r.epoch_ms
        << "," << r.samples_per_sec << "," << r.final_loss << "," << r.accuracy << "\n";
}

static void writeCsv(const std::vector<BenchmarkResult>& results, const std::string& path) {
    std::ofstream f(path);
    if (!f) return;
    f << "engine,dataset,model,samples,features,epochs,train_sec,epoch_ms,samples_per_sec,final_loss,accuracy\n";
    for (const auto& r : results) {
        f << r.engine << "," << r.dataset << ",";
        writeCsvEscaped(f, r.model);
        f << "," << r.samples << "," << r.features << "," << r.epochs << "," << r.train_sec << "," << r.epoch_ms
            << "," << r.samples_per_sec << "," << r.final_loss << "," << r.accuracy << "\n";
    }
}

const std::string BENCH_ENGINE = "akiraML3";

void runBenchmarkSuite(std::vector<BenchmarkResult>& results, bool writeToFile = true) {
    results.clear();

    // Regression sweep: vary samples for scaling graph (same arch, 10 feat, 30 epochs)
    const std::vector<int> reg_sizes = { 100, 250, 500, 1000, 2000, 5000 };
    for (int n : reg_sizes) {
        std::string name = "reg_n" + std::to_string(n);
        runRegBenchmark(BENCH_ENGINE, name, n, 10, 30, { 10, 16, 8, 1 }, results);
    }

    // Regression: fixed sizes (legacy names)
    runRegBenchmark(BENCH_ENGINE, "synthetic_reg_500", 500, 10, 30, {}, results);
    runRegBenchmark(BENCH_ENGINE, "synthetic_reg_1000", 1000, 10, 30, {}, results);
    runRegBenchmark(BENCH_ENGINE, "synthetic_reg_2000", 2000, 10, 30, {}, results);
    runRegBenchmark(BENCH_ENGINE, "synthetic_reg_5000", 5000, 10, 20, {}, results);

    // Classification: synthetic
    auto clf500 = generateClassificationData(500, 5, 3);
    std::vector<std::vector<float>> X500;
    std::vector<int> labels500;
    splitDataClassification(clf500, 5, 3, X500, labels500);
    runClfBenchmark(BENCH_ENGINE, "synthetic_clf_500", X500, labels500, 5, 3, { 5, 8, 3 }, 30, results);

    auto clf1000 = generateClassificationData(1000, 5, 3);
    std::vector<std::vector<float>> X1000;
    std::vector<int> labels1000;
    splitDataClassification(clf1000, 5, 3, X1000, labels1000);
    runClfBenchmark(BENCH_ENGINE, "synthetic_clf_1000", X1000, labels1000, 5, 3, { 5, 8, 3 }, 30, results);

    // Classification sweep: vary samples (5 feat, 3 classes)
    for (int n : { 200, 500, 1000, 2000 }) {
        auto data = generateClassificationData(n, 5, 3);
        std::vector<std::vector<float>> X;
        std::vector<int> labels;
        splitDataClassification(data, 5, 3, X, labels);
        runClfBenchmark(BENCH_ENGINE, "clf_n" + std::to_string(n), X, labels, 5, 3, { 5, 8, 3 }, 30, results);
    }

    // Iris & Wine
    std::vector<std::vector<float>> X_iris;
    std::vector<int> labels_iris;
    std::tie(X_iris, labels_iris) = loadBuiltinIris();
    normalizeFeaturesMinMax(X_iris);
    runClfBenchmark(BENCH_ENGINE, "iris", X_iris, labels_iris, 4, 3, { 4, 8, 3 }, 100, results);

    std::vector<std::vector<float>> X_wine;
    std::vector<int> labels_wine;
    std::tie(X_wine, labels_wine) = loadBuiltinWine();
    normalizeFeaturesMinMax(X_wine);
    runClfBenchmark(BENCH_ENGINE, "wine", X_wine, labels_wine, 13, 3, { 13, 16, 3 }, 50, results);

    if (writeToFile) writeCsv(results, "benchmark_results.csv");
}

int main() {
    std::vector<BenchmarkResult> results;
    runBenchmarkSuite(results, true);

    std::cout << "engine,dataset,model,samples,features,epochs,train_sec,epoch_ms,samples_per_sec,final_loss,accuracy\n";
    for (const auto& r : results) {
        printCsvRow(r);
    }
    std::cout << "\nResults written to benchmark_results.csv. Run: python scripts/plot_benchmarks.py [benchmark_results.csv ...]\n";
    return 0;
}
