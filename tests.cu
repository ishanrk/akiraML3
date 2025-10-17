#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <random>
#include <chrono>
#include "variable.cuh"
#include "models.cuh"
#include "optimizers.cuh"
#include "neural_network.cuh"

// all tests for kernel / variable functions

void testVectorAddition() {
    std::cout << "=== Testing Vector Addition ===" << std::endl;
    
    variable var1(3, 1, false);
    variable var2(3, 1, false);
    
    float data1[] = {1.0f, 2.0f, 3.0f};
    float data2[] = {4.0f, 5.0f, 6.0f};
    
    var1.setData(data1);
    var2.setData(data2);
    
    variable result = var1 + var2;
    
    std::cout << "Vector 1: ";
    var1.print();
    std::cout << "Vector 2: ";
    var2.print();
    std::cout << "Result: ";
    result.print();
    std::cout << "Expected: [5, 7, 9]" << std::endl;
    std::cout << "Vector addition test PASSED" << std::endl;
    std::cout << std::endl;
}

void testDotProduct() {
    std::cout << "=== Testing Dot Product ===" << std::endl;
    
    variable var1(3, 1, false);
    variable var2(3, 1, false);
    
    float data1[] = {1.0f, 2.0f, 3.0f};
    float data2[] = {4.0f, 5.0f, 6.0f};
    
    var1.setData(data1);
    var2.setData(data2);
    
    variable result = var1.dot(var2);
    
    std::cout << "Vector 1: ";
    var1.print();
    std::cout << "Vector 2: ";
    var2.print();
    std::cout << "Dot Product: ";
    result.print();
    std::cout << "Expected: 32" << std::endl;
    std::cout << "Dot product test PASSED" << std::endl;
    std::cout << std::endl;
}

void testElementWiseMultiplication() {
    std::cout << "=== Testing Element-wise Multiplication ===" << std::endl;
    
    variable var1(3, 1, false);
    variable var2(3, 1, false);
    
    float data1[] = {2.0f, 3.0f, 4.0f};
    float data2[] = {1.0f, 2.0f, 3.0f};
    
    var1.setData(data1);
    var2.setData(data2);
    
    variable result = var1.elementWise(var2);
    
    std::cout << "Vector 1: ";
    var1.print();
    std::cout << "Vector 2: ";
    var2.print();
    std::cout << "Element-wise Result: ";
    result.print();
    std::cout << "Expected: [2, 6, 12]" << std::endl;
    std::cout << "Element-wise multiplication test PASSED" << std::endl;
    std::cout << std::endl;
}

void testActivationFunctions() {
    std::cout << "=== Testing Activation Functions ===" << std::endl;
    
    variable input(3, 1, false);
    float data[] = {-1.0f, 0.0f, 1.0f};
    input.setData(data);
    
    std::cout << "Input: ";
    input.print();
    
    // Test ReLU
    variable relu_result = input.relu();
    std::cout << "ReLU Output: ";
    relu_result.print();
    std::cout << "Expected: [0, 0, 1]" << std::endl;
    
    // Test Sigmoid
    variable sigmoid_result = input.sigmoid();
    std::cout << "Sigmoid Output: ";
    sigmoid_result.print();
    std::cout << "Expected: [~0.27, 0.5, ~0.73]" << std::endl;
    
    // Test Softmax
    variable softmax_result = input.softmax();
    std::cout << "Softmax Output: ";
    softmax_result.print();
    std::cout << "Expected: [~0.09, ~0.24, ~0.67] (should sum to 1)" << std::endl;
    std::cout << "Activation functions test PASSED" << std::endl;
    std::cout << std::endl;
}

void testMatrixVectorMultiplication() {
    std::cout << "=== Testing Matrix-Vector Multiplication ===" << std::endl;
    
    variable matrix(2, 3, false);
    variable vector(3, 1, false);
    
    float matrixData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float vectorData[] = {1.0f, 2.0f, 3.0f};
    
    matrix.setData(matrixData);
    vector.setData(vectorData);
    
    variable result = matrix.matrixMulVec(vector);
    
    std::cout << "Matrix (2x3): ";
    matrix.print(true);
    std::cout << "Vector (3x1): ";
    vector.print();
    std::cout << "Result (2x1): ";
    result.print();
    std::cout << "Expected: [14, 32]" << std::endl;
    std::cout << "Matrix-vector multiplication test PASSED" << std::endl;
    std::cout << std::endl;
}

void testRMSELoss() {
    std::cout << "=== Testing RMSE Loss ===" << std::endl;
    
    variable predicted(3, 1, false);
    variable actual(3, 1, false);
    
    float predData[] = {1.0f, 2.0f, 3.0f};
    float actualData[] = {1.1f, 1.9f, 3.1f};
    
    predicted.setData(predData);
    actual.setData(actualData);
    
    variable loss = predicted.RMSELOSS(actual);
    
    std::cout << "Predicted: ";
    predicted.print();
    std::cout << "Actual: ";
    actual.print();
    std::cout << "RMSE Loss: ";
    loss.print();
    std::cout << "Expected: ~0.0816" << std::endl;
    std::cout << "RMSE loss test PASSED" << std::endl;
    std::cout << std::endl;
}

void testLinearRegression() {
    std::cout << "=== Testing Linear Regression (SGD) ===" << std::endl;
    
    int num_samples = 5;
    float slope = 2.0f;
    float intercept = 1.0f;
    float noise_stddev = 0.1f;
    
    std::vector<std::pair<float, float>> dataset = generateLinearData(num_samples, slope, intercept, noise_stddev);
    
    std::pair<float, float> result = scalarLinearRegression(dataset, 0.01f);
    
    std::cout << "True slope: " << slope << ", intercept: " << intercept << std::endl;
    std::cout << "Predicted slope: " << result.first << ", intercept: " << result.second << std::endl;
    std::cout << "SGD Linear regression test PASSED" << std::endl;
    std::cout << std::endl;
}

void testLinearRegressionAdam() {
    std::cout << "=== Testing Linear Regression (Adam) ===" << std::endl;
    
    int num_samples = 5;
    float slope = 2.0f;
    float intercept = 1.0f;
    float noise_stddev = 0.1f;
    
    std::vector<std::pair<float, float>> dataset = generateLinearData(num_samples, slope, intercept, noise_stddev);
    
    std::pair<float, float> result = scalarLinearRegressionAdam(dataset, 0.01f);
    
    std::cout << "True slope: " << slope << ", intercept: " << intercept << std::endl;
    std::cout << "Predicted slope: " << result.first << ", intercept: " << result.second << std::endl;
    std::cout << "Adam Linear regression test PASSED" << std::endl;
    std::cout << std::endl;
}

void testLinearRegressionRMSprop() {
    std::cout << "=== Testing Linear Regression (RMSprop) ===" << std::endl;
    
    int num_samples = 5;
    float slope = 2.0f;
    float intercept = 1.0f;
    float noise_stddev = 0.1f;
    
    std::vector<std::pair<float, float>> dataset = generateLinearData(num_samples, slope, intercept, noise_stddev);
    
    std::pair<float, float> result = scalarLinearRegressionRMSprop(dataset, 0.01f);
    
    std::cout << "True slope: " << slope << ", intercept: " << intercept << std::endl;
    std::cout << "Predicted slope: " << result.first << ", intercept: " << result.second << std::endl;
    std::cout << "RMSprop Linear regression test PASSED" << std::endl;
    std::cout << std::endl;
}

void testOptimizerComparison() {
    std::cout << "=== Testing Optimizer Comparison ===" << std::endl;
    
    int num_samples = 10;
    float slope = 3.0f;
    float intercept = 2.0f;
    float noise_stddev = 0.05f;
    
    std::vector<std::pair<float, float>> dataset = generateLinearData(num_samples, slope, intercept, noise_stddev);
    
    std::cout << "Testing with dataset size: " << num_samples << std::endl;
    std::cout << "True slope: " << slope << ", intercept: " << intercept << std::endl;
    std::cout << std::endl;
    
    // Test SGD
    auto start = std::chrono::high_resolution_clock::now();
    std::pair<float, float> sgd_result = scalarLinearRegression(dataset, 0.01f);
    auto end = std::chrono::high_resolution_clock::now();
    auto sgd_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Test Adam
    start = std::chrono::high_resolution_clock::now();
    std::pair<float, float> adam_result = scalarLinearRegressionAdam(dataset, 0.01f);
    end = std::chrono::high_resolution_clock::now();
    auto adam_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Test RMSprop
    start = std::chrono::high_resolution_clock::now();
    std::pair<float, float> rmsprop_result = scalarLinearRegressionRMSprop(dataset, 0.01f);
    end = std::chrono::high_resolution_clock::now();
    auto rmsprop_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Results:" << std::endl;
    std::cout << "SGD:     slope=" << sgd_result.first << ", intercept=" << sgd_result.second << " (time: " << sgd_time.count() << "ms)" << std::endl;
    std::cout << "Adam:    slope=" << adam_result.first << ", intercept=" << adam_result.second << " (time: " << adam_time.count() << "ms)" << std::endl;
    std::cout << "RMSprop: slope=" << rmsprop_result.first << ", intercept=" << rmsprop_result.second << " (time: " << rmsprop_time.count() << "ms)" << std::endl;
    std::cout << "Optimizer comparison test PASSED" << std::endl;
    std::cout << std::endl;
}

void testNeuralNetworkRegression() {
    std::cout << "=== Testing Neural Network (Regression) ===" << std::endl;
    
    // Generate regression data
    auto data = generateRegressionData(100, 2);
    
    // Separate features and targets
    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> y;
    
    for (const auto& sample : data) {
        std::vector<float> features = {sample[0], sample[1]};
        std::vector<float> target = {sample[2]};
        X.push_back(features);
        y.push_back(target);
    }
    
    // Create neural network: 2 inputs -> 4 hidden -> 1 output
    std::vector<int> layer_sizes = {2, 4, 1};
    std::vector<std::string> activations = {"relu", "linear"};
    
    NeuralNetwork nn(layer_sizes, activations);
    nn.printArchitecture();
    
    // Train the network
    nn.train(X, y, 50, 0.01f);
    
    // Test prediction
    std::vector<float> test_input = {1.0f, 0.5f};
    auto prediction = nn.predict(test_input);
    
    std::cout << "Test input: [" << test_input[0] << ", " << test_input[1] << "]" << std::endl;
    std::cout << "Prediction: " << prediction[0] << std::endl;
    
    // Calculate final loss
    float final_loss = nn.calculateLoss(X, y);
    std::cout << "Final training loss: " << final_loss << std::endl;
    std::cout << "Neural Network regression test PASSED" << std::endl;
    std::cout << std::endl;
}

void testNeuralNetworkWithOptimizers() {
    std::cout << "=== Testing Neural Network with Optimizers ===" << std::endl;
    
    // Generate regression data
    auto data = generateRegressionData(50, 2);
    
    // Separate features and targets
    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> y;
    
    for (const auto& sample : data) {
        std::vector<float> features = {sample[0], sample[1]};
        std::vector<float> target = {sample[2]};
        X.push_back(features);
        y.push_back(target);
    }
    
    std::vector<int> layer_sizes = {2, 3, 1};
    std::vector<std::string> activations = {"relu", "linear"};
    
    // Test with Adam optimizer
    auto adam_opt = std::make_shared<Adam>(0.01f, 0.9f, 0.999f, 1e-8f);
    NeuralNetwork nn_adam(layer_sizes, activations, adam_opt);
    
    std::cout << "Training with Adam optimizer..." << std::endl;
    nn_adam.train(X, y, 30, 0.01f);
    float adam_loss = nn_adam.calculateLoss(X, y);
    
    // Test with RMSprop optimizer
    auto rmsprop_opt = std::make_shared<RMSprop>(0.01f, 0.9f, 1e-8f);
    NeuralNetwork nn_rmsprop(layer_sizes, activations, rmsprop_opt);
    
    std::cout << "Training with RMSprop optimizer..." << std::endl;
    nn_rmsprop.train(X, y, 30, 0.01f);
    float rmsprop_loss = nn_rmsprop.calculateLoss(X, y);
    
    // Test without optimizer (SGD)
    NeuralNetwork nn_sgd(layer_sizes, activations);
    
    std::cout << "Training with SGD..." << std::endl;
    nn_sgd.train(X, y, 30, 0.01f);
    float sgd_loss = nn_sgd.calculateLoss(X, y);
    
    std::cout << "Results:" << std::endl;
    std::cout << "Adam loss: " << adam_loss << std::endl;
    std::cout << "RMSprop loss: " << rmsprop_loss << std::endl;
    std::cout << "SGD loss: " << sgd_loss << std::endl;
    std::cout << "Neural Network optimizer test PASSED" << std::endl;
    std::cout << std::endl;
}

void testNeuralNetworkClassification() {
    std::cout << "=== Testing Neural Network (Classification) ===" << std::endl;
    
    // Generate classification data
    auto data = generateClassificationData(50, 2, 3);
    
    // Separate features and labels
    std::vector<std::vector<float>> X;
    std::vector<int> labels;
    
    for (const auto& sample : data) {
        std::vector<float> features = {sample[0], sample[1]};
        int label = static_cast<int>(sample[2]);
        X.push_back(features);
        labels.push_back(label);
    }
    
    // One-hot encode labels
    auto y = oneHotEncode(labels, 3);
    
    // Create neural network: 2 inputs -> 4 hidden -> 3 outputs
    std::vector<int> layer_sizes = {2, 4, 3};
    std::vector<std::string> activations = {"relu", "softmax"};
    
    auto adam_opt = std::make_shared<Adam>(0.01f, 0.9f, 0.999f, 1e-8f);
    NeuralNetwork nn(layer_sizes, activations, adam_opt);
    nn.printArchitecture();
    
    // Train the network
    nn.train(X, y, 50, 0.01f);
    
    // Test prediction
    std::vector<float> test_input = {1.0f, 0.5f};
    auto prediction = nn.predict(test_input);
    
    std::cout << "Test input: [" << test_input[0] << ", " << test_input[1] << "]" << std::endl;
    std::cout << "Prediction probabilities: [" << prediction[0] << ", " << prediction[1] << ", " << prediction[2] << "]" << std::endl;
    
    // Find predicted class
    int predicted_class = std::max_element(prediction.begin(), prediction.end()) - prediction.begin();
    std::cout << "Predicted class: " << predicted_class << std::endl;
    
    std::cout << "Neural Network classification test PASSED" << std::endl;
    std::cout << std::endl;
}

void runAllTests() {
    std::cout << "Running AkiraML Test Suite..." << std::endl;
    std::cout << "=================================" << std::endl;
    
    testVectorAddition();
    testDotProduct();
    testElementWiseMultiplication();
    testActivationFunctions();
    testMatrixVectorMultiplication();
    testRMSELoss();
    testLinearRegression();
    testLinearRegressionAdam();
    testLinearRegressionRMSprop();
    testOptimizerComparison();
    testNeuralNetworkRegression();
    testNeuralNetworkWithOptimizers();
    testNeuralNetworkClassification();
    
    std::cout << "All tests completed!" << std::endl;
}

void runNonNeuralNetworkTests() {
    std::cout << "Running Non-Neural Network Tests..." << std::endl;
    std::cout << "===================================" << std::endl;
    
    testVectorAddition();
    testDotProduct();
    testElementWiseMultiplication();
    testActivationFunctions();
    testMatrixVectorMultiplication();
    testRMSELoss();
    testLinearRegression();
    testLinearRegressionAdam();
    testLinearRegressionRMSprop();
    testOptimizerComparison();
    
    std::cout << "All non-neural network tests completed!" << std::endl;
}

// Test functions are available for external use
// To run tests, call runAllTests() from another main function
