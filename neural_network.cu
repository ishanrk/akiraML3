#include "neural_network.cuh"
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

NeuralNetwork::NeuralNetwork(const std::vector<int>& sizes, const std::vector<std::string>& activation_funcs, 
                             std::shared_ptr<Optimizer> opt) 
    : layer_sizes(sizes), activations(activation_funcs), optimizer(opt), num_layers(sizes.size()) {
    
    if (sizes.size() < 2) {
        throw std::invalid_argument("Neural network must have at least 2 layers (input and output)");
    }
    
    if (activation_funcs.size() != sizes.size() - 1) {
        throw std::invalid_argument("Number of activation functions must be one less than number of layers");
    }
    
    initializeWeights();
    initializeBiases();
}

NeuralNetwork::~NeuralNetwork() {
    // Destructor - variables will clean up their own memory
}

void NeuralNetwork::initializeWeights() {
    weights.clear();
    
    for (int i = 0; i < num_layers - 1; i++) {
        // Xavier/Glorot initialization
        float xavier_std = sqrtf(2.0f / (layer_sizes[i] + layer_sizes[i + 1]));
        
        variable weight(layer_sizes[i + 1], layer_sizes[i], false);
        
        // Initialize with Xavier normal distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, xavier_std);
        
        for (int j = 0; j < layer_sizes[i + 1] * layer_sizes[i]; j++) {
            weight.data[j] = dist(gen);
        }
        
        weights.push_back(weight);
    }
}

void NeuralNetwork::initializeBiases() {
    biases.clear();
    
    for (int i = 1; i < num_layers; i++) {
        variable bias(layer_sizes[i], 1, false);
        
        // Initialize biases to small positive values
        for (int j = 0; j < layer_sizes[i]; j++) {
            bias.data[j] = 0.1f;
        }
        
        biases.push_back(bias);
    }
}

variable NeuralNetwork::forwardPass(const variable& input) {
    variable current = input;
    
    for (int i = 0; i < num_layers - 1; i++) {
        // Linear transformation: output = weights * input + bias
        variable linear_output = weights[i].matrixMulVec(current);
        
        // Add bias
        variable biased_output = linear_output + biases[i];
        
        // Apply activation function
        if (activations[i] == "relu") {
            current = biased_output.relu();
        } else if (activations[i] == "sigmoid") {
            current = biased_output.sigmoid();
        } else if (activations[i] == "softmax") {
            current = biased_output.softmax();
        } else {
            // Default to no activation (linear)
            current = biased_output;
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
                          int epochs, float learning_rate) {
    if (X.size() != y.size()) {
        throw std::invalid_argument("Number of input samples must match number of target samples");
    }
    
    if (X.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }
    
    int input_size = X[0].size();
    int output_size = y[0].size();
    
    if (input_size != layer_sizes[0]) {
        throw std::invalid_argument("Input size does not match network input layer size");
    }
    
    if (output_size != layer_sizes[num_layers - 1]) {
        throw std::invalid_argument("Output size does not match network output layer size");
    }
    
    std::cout << "Training neural network for " << epochs << " epochs..." << std::endl;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < X.size(); i++) {
        // Create input variable
        variable input(input_size, 1, false);
        std::vector<float> input_data = X[i];
        input.setData(input_data.data());
            
        // Create target variable
        variable target(output_size, 1, false);
        std::vector<float> target_data = y[i];
        target.setData(target_data.data());
            
            // Forward pass
            variable prediction = forwardPass(input);
            
            // Calculate loss
            variable target_copy = target;
            variable loss = prediction.RMSELOSS(target_copy);
            total_loss += *(loss.data);
            
            // Backward pass
            float* arr = { 0 };
            loss.backward(&loss, arr, 0);
            
            // Update parameters
            if (optimizer != nullptr) {
                for (int j = 0; j < weights.size(); j++) {
                    weights[j].updateWithOptimizer(epoch);
                    biases[j].updateWithOptimizer(epoch);
                }
            } else {
                // Use simple SGD if no optimizer is set
                for (int j = 0; j < weights.size(); j++) {
                    weights[j].update(learning_rate);
                    biases[j].update(learning_rate);
                }
            }
        }
        
        float avg_loss = total_loss / X.size();
        
        if (epoch % (epochs / 10) == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << "/" << epochs << ", Loss: " << avg_loss << std::endl;
        }
    }
    
    std::cout << "Training completed!" << std::endl;
}

std::vector<float> NeuralNetwork::predict(const std::vector<float>& input) {
    if (input.size() != layer_sizes[0]) {
        throw std::invalid_argument("Input size does not match network input layer size");
    }
    
    variable input_var(input.size(), 1, false);
    std::vector<float> input_data = input;
    input_var.setData(input_data.data());
    
    variable prediction = forwardPass(input_var);
    
    std::vector<float> result(layer_sizes[num_layers - 1]);
    for (int i = 0; i < layer_sizes[num_layers - 1]; i++) {
        result[i] = prediction.data[i];
    }
    
    return result;
}

std::vector<std::vector<float>> NeuralNetwork::predictBatch(const std::vector<std::vector<float>>& inputs) {
    std::vector<std::vector<float>> results;
    
    for (const auto& input : inputs) {
        results.push_back(predict(input));
    }
    
    return results;
}

float NeuralNetwork::calculateLoss(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& y) {
    if (X.size() != y.size()) {
        throw std::invalid_argument("Number of input samples must match number of target samples");
    }
    
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < X.size(); i++) {
        variable input(X[i].size(), 1, false);
        std::vector<float> input_data = X[i];
        input.setData(input_data.data());
        
        variable target(y[i].size(), 1, false);
        std::vector<float> target_data = y[i];
        target.setData(target_data.data());
        
        variable prediction = forwardPass(input);
        variable target_copy = target;
        variable loss = prediction.RMSELOSS(target_copy);
        
        total_loss += *(loss.data);
    }
    
    return total_loss / X.size();
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

// Utility functions
std::vector<std::vector<float>> generateClassificationData(int num_samples, int num_features, int num_classes) {
    std::vector<std::vector<float>> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> feature_dist(-2.0f, 2.0f);
    std::uniform_int_distribution<int> class_dist(0, num_classes - 1);
    
    for (int i = 0; i < num_samples; i++) {
        std::vector<float> sample;
        
        // Generate features
        for (int j = 0; j < num_features; j++) {
            sample.push_back(feature_dist(gen));
        }
        
        // Generate class label
        int class_label = class_dist(gen);
        sample.push_back(static_cast<float>(class_label));
        
        data.push_back(sample);
    }
    
    return data;
}

std::vector<std::vector<float>> generateRegressionData(int num_samples, int num_features) {
    std::vector<std::vector<float>> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> feature_dist(-2.0f, 2.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.1f);
    
    // Generate random weights for the true function
    std::vector<float> true_weights(num_features);
    for (int i = 0; i < num_features; i++) {
        true_weights[i] = feature_dist(gen);
    }
    
    for (int i = 0; i < num_samples; i++) {
        std::vector<float> sample;
        
        // Generate features
        for (int j = 0; j < num_features; j++) {
            sample.push_back(feature_dist(gen));
        }
        
        // Generate target using linear combination + noise
        float target = 0.0f;
        for (int j = 0; j < num_features; j++) {
            target += sample[j] * true_weights[j];
        }
        target += noise_dist(gen);
        
        sample.push_back(target);
        data.push_back(sample);
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
