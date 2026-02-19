#include "neural_network.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>

static Act parseAct(const std::string& s) {
    if (s == "relu") return Act::Relu;
    if (s == "sigmoid") return Act::Sigmoid;
    if (s == "softmax") return Act::Softmax;
    return Act::Linear;
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& sizes, const std::vector<std::string>& activation_funcs,
                             std::shared_ptr<Optimizer> opt)
    : layer_sizes(sizes), activations(activation_funcs), optimizer(opt), num_layers(static_cast<int>(sizes.size())) {
    if (sizes.size() < 2)
        throw std::invalid_argument("Neural network must have at least 2 layers (input and output)");
    if (activation_funcs.size() != sizes.size() - 1)
        throw std::invalid_argument("Number of activation functions must be one less than number of layers");
    act_type_.reserve(activation_funcs.size());
    for (const auto& a : activation_funcs)
        act_type_.push_back(parseAct(a));
    initializeWeights();
    initializeBiases();
    if (optimizer) setOptimizer(optimizer);
}

NeuralNetwork::~NeuralNetwork() {
    // Destructor - variables will clean up their own memory
}

void NeuralNetwork::initializeWeights() {
    weights.clear();
    weights.reserve(static_cast<size_t>(num_layers - 1));
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = 0; i < num_layers - 1; i++) {
        const int out = layer_sizes[i + 1], in = layer_sizes[i];
        const float xavier_std = sqrtf(2.0f / (in + out));
        variable weight(out, in, false);
        std::normal_distribution<float> dist(0.0f, xavier_std);
        const int n = out * in;
        for (int j = 0; j < n; j++)
            weight.data[j] = dist(gen);
        weights.push_back(std::move(weight));
    }
}

void NeuralNetwork::initializeBiases() {
    biases.clear();
    biases.reserve(static_cast<size_t>(num_layers - 1));
    for (int i = 1; i < num_layers; i++) {
        variable bias(layer_sizes[i], 1, false);
        const int n = layer_sizes[i];
        for (int j = 0; j < n; j++)
            bias.data[j] = 0.1f;
        biases.push_back(std::move(bias));
    }
}

variable NeuralNetwork::forwardPass(const variable& input) {
    variable current = input;
    for (int i = 0; i < num_layers - 1; i++) {
        variable linear_output = weights[i].matrixMulVec(current);
        variable biased_output = linear_output + biases[i];
        switch (act_type_[i]) {
            case Act::Relu:    current = biased_output.relu(); break;
            case Act::Sigmoid: current = biased_output.sigmoid(); break;
            case Act::Softmax: current = biased_output.softmax(); break;
            default:           current = biased_output; break;
        }
    }
    return current;
}

void NeuralNetwork::backwardPass(const variable& input, const variable& target) {
    // Forward pass to get predictions
    variable prediction = forwardPass(input);
    
    // Create a non-const copy of target for RMSELOSS
    variable target_copy = target;
    
    // Calculate loss (using RMSE for regression, could be extended for classification)
    variable loss = prediction.RMSELOSS(target_copy);
    
    // Backward pass
    float* arr = { 0 };
    loss.backward(&loss, arr, 0);
    
    // Update parameters using optimizer
    if (optimizer != nullptr) {
        for (int i = 0; i < weights.size(); i++) {
            weights[i].updateWithOptimizer(0); // Iteration number could be passed
            biases[i].updateWithOptimizer(0);
        }
    }
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& y,
                          int epochs, float learning_rate, bool verbose) {
    const size_t n_samples = X.size();
    if (n_samples != y.size())
        throw std::invalid_argument("Number of input samples must match number of target samples");
    if (n_samples == 0)
        throw std::invalid_argument("Training data cannot be empty");
    const int input_size = static_cast<int>(X[0].size());
    const int output_size = static_cast<int>(y[0].size());
    if (input_size != layer_sizes[0])
        throw std::invalid_argument("Input size does not match network input layer size");
    if (output_size != layer_sizes[num_layers - 1])
        throw std::invalid_argument("Output size does not match network output layer size");

    std::vector<float> X_flat(n_samples * input_size);
    std::vector<float> y_flat(n_samples * output_size);
    for (size_t i = 0; i < n_samples; i++) {
        std::memcpy(X_flat.data() + i * input_size, X[i].data(), input_size * sizeof(float));
        std::memcpy(y_flat.data() + i * output_size, y[i].data(), output_size * sizeof(float));
    }
    const float* X_ptr = X_flat.data();
    const float* y_ptr = y_flat.data();

    if (verbose) std::cout << "Training neural network for " << epochs << " epochs..." << std::endl;

    const size_t n_weights = weights.size();
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < n_samples; i++) {
            variable input(input_size, 1, false);
            input.setData(const_cast<float*>(X_ptr + i * input_size));
            variable target(output_size, 1, false);
            target.setData(const_cast<float*>(y_ptr + i * output_size));
            variable prediction = forwardPass(input);
            variable target_ref = target;
            variable loss = prediction.RMSELOSS(target_ref);
            total_loss += *loss.data;
            float* arr = nullptr;
            loss.backward(&loss, arr, 0);
            if (optimizer) {
                for (size_t j = 0; j < n_weights; j++) {
                    weights[j].updateWithOptimizer(epoch);
                    biases[j].updateWithOptimizer(epoch);
                }
            } else {
                for (size_t j = 0; j < n_weights; j++) {
                    weights[j].update(learning_rate);
                    biases[j].update(learning_rate);
                }
            }
        }
        const float avg_loss = total_loss / static_cast<float>(n_samples);
        if (verbose && (epochs <= 10 || epoch % (epochs / 10) == 0 || epoch == epochs - 1))
            std::cout << "Epoch " << epoch << "/" << epochs << ", Loss: " << avg_loss << std::endl;
    }
    if (verbose) std::cout << "Training completed!" << std::endl;
}

std::vector<float> NeuralNetwork::predict(const std::vector<float>& input) {
    if (input.size() != static_cast<size_t>(layer_sizes[0])) {
        throw std::invalid_argument("Input size does not match network input layer size");
    }
    variable input_var(static_cast<int>(input.size()), 1, false);
    input_var.setData(const_cast<float*>(input.data()));
    variable prediction = forwardPass(input_var);
    std::vector<float> result(layer_sizes[num_layers - 1]);
    for (int i = 0; i < layer_sizes[num_layers - 1]; i++)
        result[i] = prediction.data[i];
    return result;
}

std::vector<std::vector<float>> NeuralNetwork::predictBatch(const std::vector<std::vector<float>>& inputs) {
    std::vector<std::vector<float>> results;
    results.reserve(inputs.size());
    for (const auto& input : inputs)
        results.push_back(predict(input));
    return results;
}

float NeuralNetwork::calculateLoss(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& y) {
    if (X.size() != y.size())
        throw std::invalid_argument("Number of input samples must match number of target samples");
    const size_t n = X.size();
    if (n == 0) return 0.0f;
    float total_loss = 0.0f;
    const int in_sz = layer_sizes[0];
    const int out_sz = layer_sizes[num_layers - 1];
    for (size_t i = 0; i < n; i++) {
        variable input(in_sz, 1, false);
        input.setData(const_cast<float*>(X[i].data()));
        variable target(out_sz, 1, false);
        target.setData(const_cast<float*>(y[i].data()));
        variable prediction = forwardPass(input);
        variable target_ref = target;
        variable loss = prediction.RMSELOSS(target_ref);
        total_loss += *loss.data;
    }
    return total_loss / static_cast<float>(n);
}

void NeuralNetwork::printArchitecture() {
    std::cout << "Neural Network Architecture:" << std::endl;
    std::cout << "============================" << std::endl;
    
    for (int i = 0; i < num_layers; i++) {
        std::cout << "Layer " << i << ": " << layer_sizes[i] << " neurons";
        
        if (i < num_layers - 1) {
            std::cout << " -> " << activations[i] << " activation";
        }
        
        std::cout << std::endl;
    }
    
    std::cout << "Total parameters: ";
    int total_params = 0;
    for (int i = 0; i < num_layers - 1; i++) {
        total_params += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1];
    }
    std::cout << total_params << std::endl;
    std::cout << std::endl;
}

void NeuralNetwork::setOptimizer(std::shared_ptr<Optimizer> opt) {
    optimizer = opt;
    
    // Set optimizer for all parameters
    for (int i = 0; i < weights.size(); i++) {
        weights[i].setOptimizer(opt);
        biases[i].setOptimizer(opt);
    }
}

void NeuralNetwork::save(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("NeuralNetwork::save: cannot open " + path);
    f << "MLP\nLAYERS\n" << num_layers << "\n";
    for (int i = 0; i < num_layers; i++) f << layer_sizes[i] << "\n";
    f << "ACTIVATIONS\n" << activations.size() << "\n";
    for (size_t i = 0; i < activations.size(); i++) f << activations[i] << "\n";
    for (int i = 0; i < num_layers - 1; i++) {
        f << "WEIGHT\n" << i << "\n" << weights[i].dim1 << " " << weights[i].dim2 << "\n";
        int n = weights[i].dim1 * weights[i].dim2;
        for (int j = 0; j < n; j++) f << weights[i].data[j] << "\n";
    }
    for (int i = 0; i < num_layers - 1; i++) {
        f << "BIAS\n" << i << "\n" << biases[i].dim1 << "\n";
        for (int j = 0; j < biases[i].dim1; j++) f << biases[i].data[j] << "\n";
    }
    f << "END\n";
}

NeuralNetwork NeuralNetwork::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("NeuralNetwork::load: cannot open " + path);
    std::string tag;
    f >> tag;
    if (tag != "MLP") throw std::runtime_error("NeuralNetwork::load: expected MLP, got " + tag);
    f >> tag;
    if (tag != "LAYERS") throw std::runtime_error("NeuralNetwork::load: expected LAYERS");
    int nl;
    f >> nl;
    std::vector<int> sizes(nl);
    for (int i = 0; i < nl; i++) f >> sizes[i];
    f >> tag;
    if (tag != "ACTIVATIONS") throw std::runtime_error("NeuralNetwork::load: expected ACTIVATIONS");
    int na;
    f >> na;
    std::vector<std::string> acts(na);
    for (int i = 0; i < na; i++) f >> acts[i];
    NeuralNetwork nn(sizes, acts, nullptr);
    for (int i = 0; i < nl - 1; i++) {
        f >> tag;
        if (tag != "WEIGHT") throw std::runtime_error("NeuralNetwork::load: expected WEIGHT");
        int idx, rows, cols;
        f >> idx >> rows >> cols;
        int n = rows * cols;
        for (int j = 0; j < n; j++) f >> nn.weights[i].data[j];
    }
    for (int i = 0; i < nl - 1; i++) {
        f >> tag;
        if (tag != "BIAS") throw std::runtime_error("NeuralNetwork::load: expected BIAS");
        int idx, dim;
        f >> idx >> dim;
        for (int j = 0; j < dim; j++) f >> nn.biases[i].data[j];
    }
    f >> tag;
    if (tag != "END") throw std::runtime_error("NeuralNetwork::load: expected END");
    return nn;
}

// Utility functions
std::vector<std::vector<float>> generateClassificationData(int num_samples, int num_features, int num_classes) {
    std::vector<std::vector<float>> data;
    data.reserve(static_cast<size_t>(num_samples));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> feature_dist(-2.0f, 2.0f);
    std::uniform_int_distribution<int> class_dist(0, num_classes - 1);
    for (int i = 0; i < num_samples; i++) {
        std::vector<float> sample;
        sample.reserve(static_cast<size_t>(num_features) + 1u);
        for (int j = 0; j < num_features; j++)
            sample.push_back(feature_dist(gen));
        sample.push_back(static_cast<float>(class_dist(gen)));
        data.push_back(std::move(sample));
    }
    return data;
}

std::vector<std::vector<float>> generateRegressionData(int num_samples, int num_features) {
    std::vector<std::vector<float>> data;
    data.reserve(static_cast<size_t>(num_samples));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> feature_dist(-2.0f, 2.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.1f);
    std::vector<float> true_weights(static_cast<size_t>(num_features));
    for (int i = 0; i < num_features; i++)
        true_weights[i] = feature_dist(gen);
    for (int i = 0; i < num_samples; i++) {
        std::vector<float> sample;
        sample.reserve(static_cast<size_t>(num_features) + 1u);
        for (int j = 0; j < num_features; j++)
            sample.push_back(feature_dist(gen));
        float target = 0.0f;
        for (int j = 0; j < num_features; j++)
            target += sample[j] * true_weights[j];
        target += noise_dist(gen);
        sample.push_back(target);
        data.push_back(std::move(sample));
    }
    return data;
}

std::vector<std::vector<float>> oneHotEncode(const std::vector<int>& labels, int num_classes) {
    std::vector<std::vector<float>> encoded;
    
    for (int label : labels) {
        std::vector<float> one_hot(num_classes, 0.0f);
        if (label >= 0 && label < num_classes) {
            one_hot[label] = 1.0f;
        }
        encoded.push_back(one_hot);
    }
    
    return encoded;
}
