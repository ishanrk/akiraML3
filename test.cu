#include <iostream>
#include "variable.cuh"
using namespace std;

int main()
{
    variable w(1, 1, false);
    w.data[0] = 4;
    
    variable x(1, 1, false);
    x.data[0] = 3;

    variable b(1, 1, false);
    b.data[0] = 5;

    variable intermed = w.dot(x);
    variable layer = intermed + b;
    variable trueOuput(1, 1, false);
    trueOuput.data[0] = 20;
    variable loss = layer.RMSELOSS(trueOuput);
    float* rand = { 0 };
    loss.backward(&loss, rand,0);

    cout << *(loss.data) << endl;
    cout << *(layer.data) << endl;
    cout << *(intermed.data) << endl;
    cout << *(w.data) << endl;
    cout << *(b.data) << endl;

    cout << *(loss.backwardGrad) << endl;
    cout << *(layer.backwardGrad) << endl;
    cout << *(intermed.backwardGrad) << endl;
    cout << *(w.backwardGrad) << endl;
    cout << *(b.backwardGrad) << endl;

    variable weights[] = { w };
    w.update(0.1);
    cout << *(w.data) << endl;

	return 0;
}