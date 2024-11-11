#include <iostream>
#include "variable.cuh"
using namespace std;

int main()
{
	variable a(5);
	variable b(5);
	float arr[5] = { 1,2,3,4,5 };
	float arr2[5] = { 1,2,3,4,5 };

	b.setData(arr, 5);
	a.setData(arr2, 5);
	variable c = a.dot(b);
	c.print();


	return 0;
}