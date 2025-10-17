#pragma once
#include "variable.cuh"
#include "optimizers.cuh"
#include <vector>
#include <string>

class NeuralNetwork {
private:
    std::vector<int> layer_sizes;
    std::vector<variable> weights;
    std::vector<variable> biases;
    std::vector<std::string> activations;
    std::shared_ptr<Optimizer> optimizer;
    int num_layers;
    
    // Helper functions
    void initializeWeights();
    void initializeBiases();
    variable forwardPass(const variable& input);
    void backwardPass(const variable& input, const variable& target);
    
public:
    // Constructor
    NeuralNetwork(const std::vector<int>& sizes, const std::vector<std::string>& activation_funcs, 
                  std::shared_ptr<Optimizer> opt = nullptr);
    
    // Destructor
    ~NeuralNetwork();
    
    // Training and prediction
    void train(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& y, 
               int epochs, float learning_rate = 0.001f);
    std::vector<float> predict(const std::vector<float>& input);
    std::vector<std::vector<float>> predictBatch(const std::vector<std::vector<float>>& inputs);
    
    // Loss calculation
    float calculateLoss(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& y);
    
    // Utility functions
    void printArchitecture();
    void setOptimizer(std::shared_ptr<Optimizer> opt);
    
    // Getters
    std::vector<int> getLayerSizes() const { return layer_sizes; }
    int getNumLayers() const { return num_layers; }
    std::vector<std::string> getActivations() const { return activations; }
};

// Utility functions for neural network
std::vector<std::vector<float>> generateClassificationData(int num_samples, int num_features, int num_classes);
std::vector<std::vector<float>> generateRegressionData(int num_samples, int num_features);
std::vector<std::vector<float>> oneHotEncode(const std::vector<int>& labels, int num_classes);
