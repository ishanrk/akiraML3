#include <iostream>
#include "tests.cuh"

using namespace std;

int main()
{
    cout << "=== AkiraML3 Test Suite ===" << std::endl;
    cout << "Running non-neural network tests..." << std::endl;
    cout << std::endl;
    
    try {
        runNonNeuralNetworkTests();
        cout << "=== All Tests Complete ===" << endl;
    } catch (const std::exception& e) {
        cout << "Error running tests: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}