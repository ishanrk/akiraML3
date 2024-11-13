#include <iostream>
#include "models.cuh"

using namespace std;

int main()
{
    int num_samples = 10;
    float slope = 2.5;      // Specify slope (m)
    float intercept = 1.0;  // Specify intercept (c)
    float noise_stddev = 0.00001; // Standard deviation of noise

    // Generate the dataset
    std::vector<std::pair<float, float>> dataset = generateLinearData(num_samples, slope, intercept, noise_stddev);

    pair<float, float> answer = scalarLinearRegression(dataset, 0.01);
    cout << answer.first << " " <<answer.second << std::endl;
	return 0;
}