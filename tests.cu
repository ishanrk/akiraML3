#include <iostream>
#include "variable.cuh"
// all tests for kernel / variable functions


void testElemWise()
{
    variable var1(3, 1, false, {}); // Creates a 3x1 variable
    variable var2(3, 1, false, {}); // Creates a 3x1 variable

    // Allocate data for the two variables (manually filling them for simplicity)
    float data1[] = { 1.0f, 2.0f, 3.0f }; // Set data for var1
    float data2[] = { 4.0f, 5.0f, 6.0f }; // Set data for var2

    // Set data in each variable
    var1.setData(data1);
    var2.setData(data2);

    // Perform element-wise multiplication
    variable result = var1.elementWise(var2);

    // Print result data
    std::cout << "Element-wise multiplication result: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << result.data[i] << " ";
    }
    std::cout << std::endl;

    // Print gradients (gradients should just be copies of the other vector in this case)
    std::cout << "Gradient w.r.t. var1 (gradientChild1): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << result.gradientChild1[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Gradient w.r.t. var2 (gradientChild2): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << result.gradientChild2[i] << " ";
    }
    std::cout << std::endl;

}