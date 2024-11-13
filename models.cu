#include "models.cuh"
using namespace std;
std::pair<float, float> scalarLinearRegression(std::vector<std::pair<float, float>> dataset, float learningRate)
{
	float* data = (float*)malloc(dataset.size()* sizeof(float));
	float* target = (float*)malloc(dataset.size() * sizeof(float));

	for (int x = 0; x < dataset.size();x++)
	{
		data[x] = dataset[x].first;
		target[x] = dataset[x].second;	
	}

	variable X(dataset.size(), 1, false);
	X.setData(data);

	variable Y(dataset.size(), 1, false);
	Y.setData(target);

	variable theta(dataset.size(), 1, true);
	variable bias(dataset.size(), 1, true);
	
	variable parameters[2] = { theta, bias };
	variable prod(dataset.size(), 1, false);
	variable layer(dataset.size(), 1, false);
	variable loss(1, 1, false);
	cout << *(theta.data) << endl;
	cout << *(X.data) << endl;
	for (int x = 0; x < 5000; x++)
	{
		prod = X.elementWise(theta);
		
		layer = prod + bias;
		loss = layer.RMSELOSS(Y);
	
		float *arr = { 0 };
		loss.backward(&loss, arr, 0);
		
		float derivTheta = 0;
		float derivBias = 0;
		for (int x = 0; x < dataset.size();x++)
		{
			derivTheta += theta.backwardGrad[x];
		}
		for (int x = 0; x < dataset.size();x++)
		{
			theta.data[x] = theta.data[x] - learningRate * derivTheta;
		}
		for (int x = 0; x < dataset.size();x++)
		{
			derivBias += bias.backwardGrad[x];
		}
		for (int x = 0; x < dataset.size();x++)
		{
			bias.data[x] = bias.data[x] - learningRate * derivBias;
		}
	}

	return std::make_pair(theta.data[0], bias.data[0]);
}