#pragma once
#include<vector>
#include<cuda_runtime.h>
#include "kernel.cuh"
#include<iostream>

class variable
{

public:
	float* data;
	float* gradientChild1;
	float* gradientChild2;
	int dim1;
	int dim2;
	bool rand = true;
	std::vector<variable*> children;


	variable(int dimension1, int dimension2 = 1, bool random = true, std::vector<variable*>currChildren = {});

	~variable();

	variable operator+(const variable& other) const;

	variable dot(const variable& other) const;

	void print(bool matrix = false) const;

	int setData(float* arr);

	variable variable::matrixMulVec(const variable& other) const;

	void getChildren();

	void tester();

	variable variable::sigmoid() const;
	variable variable::softmax() const;
	variable variable::relu() const;
};



