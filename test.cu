#include <iostream>
#include "tests.cuh"

int main() {
    std::cout << "=== AkiraML3 Test Suite ===" << std::endl;
    std::cout << std::endl;
    try {
        std::cout << "Phase 1: Unit tests (variable, kernel, optimizers, linear regression)..." << std::endl;
        runNonNeuralNetworkTests();
        std::cout << std::endl;
        std::cout << "Phase 2: Neural network unit tests (regression, optimizers, classification)..." << std::endl;
        testNeuralNetworkRegression();
        testNeuralNetworkWithOptimizers();
        testNeuralNetworkClassification();
        std::cout << std::endl;
        std::cout << "Phase 3: Validation suite (diverse samples, verbose logs, timings)..." << std::endl;
        runValidationSuite();
        std::cout << "=== All Tests Complete ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}