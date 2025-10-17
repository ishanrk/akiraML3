#include <iostream>
#include <vector>
#include <map>
#include "variable.cuh"
std::pair<float, float> scalarLinearRegression(std::vector<std::pair<float, float>> dataset, float learningRate);
std::pair<float, float> scalarLinearRegressionAdam(std::vector<std::pair<float, float>> dataset, float learningRate);
std::pair<float, float> scalarLinearRegressionRMSprop(std::vector<std::pair<float, float>> dataset, float learningRate);

