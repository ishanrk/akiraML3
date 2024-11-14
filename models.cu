#include "models.cuh"
using namespace std;
std::pair<float, float> scalarLinearRegression(std::vector<std::pair<float, float>> dataset, float learningRate)
{
	float* data = (float*)malloc(dataset.size()* sizeof(float));
	float* target = (float*)malloc(dataset.size() * sizeof(float));
	float*rand = (float*)malloc(dataset.size() * sizeof(float));
	for (int x = 0; x < dataset.size();x++)
	{
		data[x] = dataset[x].first;
		target[x] = dataset[x].second;	
		rand[x] = 1;
	}

	variable X(dataset.size(), 1, false);
	X.setData(data);

	variable Y(dataset.size(), 1, false);
	Y.setData(target);

	variable theta(1, 1, true);
	variable bias(1, 1, true);
	variable biasVec(dataset.size(), 1, false);
	variable biasVector(dataset.size(), 1, false);
	biasVec.setData(rand);
	
	variable parameters[2] = { theta, bias };
	variable prod(dataset.size(), 1, false);
	variable layer(dataset.size(), 1, false);
	variable loss(1, 1, false);

	for (int x = 0; x < 1000; x++)
	{
		prod = X.scale(theta);
		biasVector = biasVec.scale(bias);
		layer = prod + biasVector;
		loss = layer.RMSELOSS(Y);
	
		float *arr = { 0 };
		
		loss.reverseMode(arr, 0);
		
		float derivTheta = 0;
		float derivBias = 0;
		theta.data[x] = theta.data[x] - learningRate * derivTheta;
		bias.data[x] = bias.data[x] - learningRate * derivBias;
		
		loss.print();
		std::cout << *(theta.data) << " " << *(theta.data) << std::endl;
	}

	return std::make_pair(theta.data[0], bias.data[0]);
}