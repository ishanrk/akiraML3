#pragma once
#include<vector>
#include<cuda_runtime.h>
#include "kernel.cu"

class variable
{
	float* data;
	float* gradient;
	int dim1;
	int dim2;
	bool rand = true;
	std::vector<variable*> children;

public:

	variable(int dimension1, int dimension2 = 1, bool random = true, std::vector<variable*>currChildren = {});

	~variable();

	variable operator+(const variable& other) const;

	void print() const;

	int setData(float* arr, int dimension1);
};

