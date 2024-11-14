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
	
	float* backwardGrad;
	int dim1;
	int dim2;
	bool rand = true;
	std::vector<variable*> children;
	std::vector<variable> parents = {};
	int opID;
	bool matrix;


	variable(int dimension1, int dimension2 = 1, bool random = true, std::vector<variable*>currChildren = {});

	~variable();

	variable operator+( variable& other) ;

	variable dot( variable& other) ;

	void print(bool matrix = false) const;

	int setData(float* arr);

	variable matrixMulVec( variable& other) ;

	void getChildren();

	void tester();

	variable variable::sigmoid() ;
	variable variable::softmax() ;
	variable variable::relu() ;

	

	variable RMSELOSS(variable& trueOutput);

	variable elementWise(variable& other);

	variable scale(variable& scalar);


	void reverseMode(float* gradAccum, int childID);

	void update(float lr);
};




