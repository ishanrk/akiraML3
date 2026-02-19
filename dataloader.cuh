#pragma once

#include <vector>
#include <string>
#include <utility>

// Load CSV file (comma-separated), skip empty lines, return rows as vector of float vectors
std::vector<std::vector<float>> loadCSV(const std::string& path);

// Regression: X = first num_features cols, y = last col as 1-dim vectors
void splitDataRegression(const std::vector<std::vector<float>>& data, int num_features,
                         std::vector<std::vector<float>>& X, std::vector<std::vector<float>>& y);

// Classification: X = first num_features cols, labels = last col as int (0..num_classes-1)
void splitDataClassification(const std::vector<std::vector<float>>& data, int num_features,
                             int num_classes, std::vector<std::vector<float>>& X,
                             std::vector<int>& labels);

// In-place min-max normalization per column to [0,1]
void normalizeFeaturesMinMax(std::vector<std::vector<float>>& X);

// Built-in Iris: 150 samples, 4 features, 3 classes (setosa=0, versicolor=1, virginica=2)
std::pair<std::vector<std::vector<float>>, std::vector<int>> loadBuiltinIris();

// Built-in Wine: 178 samples, 13 features, 3 classes (1->0, 2->1, 3->2)
std::pair<std::vector<std::vector<float>>, std::vector<int>> loadBuiltinWine();
