#include <iostream>
#include "models.cuh"

using namespace std;

int main()
{
    int num_samples = 100;
    float slope = 1000;      // Specify slope (m)
    float intercept = -3000;  // Specify intercept (c)
    float noise_stddev = 0.00001; // Standard deviation of noise

    // Generate the dataset
    std::vector<std::pair<float, float>> dataset = generateLinearData(num_samples, slope, intercept, noise_stddev);

    pair<float, float> answer = scalarLinearRegression(dataset,0.01 );
    cout << answer.first << " " << answer.second << std::endl;
    variable a(1, 1, true);
    a.tester();
	return 0;
}