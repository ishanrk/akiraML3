/**
 * Train an MLP on MNIST or Fashion-MNIST (CSV format).
 * Run scripts/download_mnist.py first to create data/mnist_*.csv and data/fashion_mnist_*.csv.
 * Build: nvcc -std=c++17 -O2 -o build/train_mnist train_mnist.cu neural_network.cu variable.cu kernel.cu dataloader.cu
 */
#include "neural_network.cuh"
#include "optimizers.cuh"
#include "dataloader.cuh"
#include <iostream>
#include <chrono>
#include <string>
#include <algorithm>

static void normalizePixels(std::vector<std::vector<float>>& X) {
    for (auto& row : X) {
        for (float& v : row)
            v = v / 255.0f;
    }
}

static float computeAccuracy(NeuralNetwork& nn,
    const std::vector<std::vector<float>>& X, const std::vector<int>& labels) {
    if (X.empty() || labels.empty()) return 0.0f;
    int correct = 0;
    for (size_t i = 0; i < X.size(); i++) {
        auto pred = nn.predict(X[i]);
        int c = static_cast<int>(std::max_element(pred.begin(), pred.end()) - pred.begin());
        if (c == labels[i]) correct++;
    }
    return static_cast<float>(correct) / static_cast<float>(labels.size());
}

int main(int argc, char** argv) {
    std::string dataset = "mnist";
    int epochs = 10;
    std::string train_path = "data/mnist_train.csv";
    std::string test_path = "data/mnist_test.csv";
    std::string save_path = "data/mnist_mlp.akira";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--fashion" || arg == "-f") {
            dataset = "fashion";
            train_path = "data/fashion_mnist_train.csv";
            test_path = "data/fashion_mnist_test.csv";
            save_path = "data/fashion_mnist_mlp.akira";
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::atoi(argv[++i]);
        } else if (arg == "--train" && i + 1 < argc) {
            train_path = argv[++i];
        } else if (arg == "--test" && i + 1 < argc) {
            test_path = argv[++i];
        } else if (arg == "--save" && i + 1 < argc) {
            save_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: train_mnist [--fashion] [--epochs N] [--train path] [--test path] [--save path]\n"
                      << "  --fashion    Use Fashion-MNIST (default: MNIST)\n"
                      << "  --epochs N   Train for N epochs (default 10)\n"
                      << "  --train path Train CSV (785 cols: 784 pixels + label)\n"
                      << "  --test path  Test CSV\n"
                      << "  --save path  Save model path\n"
                      << "First run: python scripts/download_mnist.py\n";
            return 0;
        }
    }

    const int num_features = 784;
    const int num_classes = 10;

    std::cout << "Loading " << train_path << " ..." << std::endl;
    std::vector<std::vector<float>> data_train = loadCSV(train_path);
    if (data_train.empty()) {
        std::cerr << "No data. Run: python scripts/download_mnist.py" << std::endl;
        return 1;
    }
    std::vector<std::vector<float>> X_train;
    std::vector<int> labels_train;
    splitDataClassification(data_train, num_features, num_classes, X_train, labels_train);
    normalizePixels(X_train);
    std::cout << "Train samples: " << X_train.size() << std::endl;

    std::cout << "Loading " << test_path << " ..." << std::endl;
    std::vector<std::vector<float>> data_test = loadCSV(test_path);
    std::vector<std::vector<float>> X_test;
    std::vector<int> labels_test;
    if (!data_test.empty()) {
        splitDataClassification(data_test, num_features, num_classes, X_test, labels_test);
        normalizePixels(X_test);
        std::cout << "Test samples: " << X_test.size() << std::endl;
    }

    std::vector<int> arch = { 784, 256, 128, 10 };
    std::vector<std::string> acts = { "relu", "relu", "softmax" };
    auto opt = std::make_shared<Adam>(0.001f, 0.9f, 0.999f, 1e-8f);
    NeuralNetwork nn(arch, acts, opt);
    nn.printArchitecture();

    std::cout << "Training for " << epochs << " epochs..." << std::endl;
    auto y_train = oneHotEncode(labels_train, num_classes);
    auto t0 = std::chrono::high_resolution_clock::now();
    nn.train(X_train, y_train, epochs, 0.001f, true);
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Training time: " << sec << " s" << std::endl;

    float train_acc = computeAccuracy(nn, X_train, labels_train);
    std::cout << "Train accuracy: " << (train_acc * 100.0f) << "%" << std::endl;
    if (!X_test.empty()) {
        float test_acc = computeAccuracy(nn, X_test, labels_test);
        std::cout << "Test accuracy:  " << (test_acc * 100.0f) << "%" << std::endl;
    }

    nn.save(save_path);
    std::cout << "Model saved to " << save_path << std::endl;
    return 0;
}
