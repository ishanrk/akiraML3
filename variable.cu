#include "variable.cuh"

variable::variable(int dimension1, int dimension2, bool random, std::vector<variable*>currChildren)
{
    dim1 = dimension1;
    dim2 = dimension2;
    rand = random;
    children = currChildren;
    data = (float*)malloc(dim1 * dim2 * sizeof(float));
    gradientChild1 = (float*)malloc(dim1 * dim2 * sizeof(float));
    if (currChildren.empty()) { children.push_back(this); }
    gradC11 = dim1;
    gradC12 = dim2;
    std::fill(gradientChild1, gradientChild1 + dim1, 1.0f);

    if (rand)
    {
        
            
           
            random_init(data, dim1, dim2);
            // Set all values in `gradient` to 0
           
        
    }
}

int variable::setData(float* arr)
{

        for (int x = 0; x < this->dim1*this->dim2;x++)
        {
            std::cout << arr[x] << std::endl;
            this->data[x] = arr[x];
        }
    
    return 1;
}

variable::~variable()
{
  
}
variable variable::operator+( variable& other)  {
    // Check if dimensions match
    if (this->dim1 != other.dim1 || this->dim2 != other.dim2) {
        throw std::invalid_argument("Dimensions must match for addition.");
    }

    // Create a new variable to store the result with dim1 and dim2 dimensions
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    temp.push_back(const_cast<variable*>(&other));
    variable result(this->dim1, this->dim2, false, temp);

    // Allocate memory for `result` data and gradient on the GPU using cudaMallocManaged
    result.data = (float*)malloc(dim1 * sizeof(float));
    result.gradientChild1 = (float*)malloc(dim1 * sizeof(float));
    result.gradientChild2 = (float*)malloc(dim1 * sizeof(float));
    result.gradC11 = dim1;
    result.gradC12 = dim2;
    result.gradC21 = dim1;
    result.gradC22 = dim2;
    // Perform element-wise addition
    
    result.data = addWithCuda(result.data, this->data, other.data, dim1);
    
    // Set `gradient` of the new variable to 0
    std::fill(result.gradientChild1, result.gradientChild1 + dim1, 1.0f);
    std::fill(result.gradientChild2, result.gradientChild2 + dim1, 1.0f);

    // Add both operands as children to the result
    this->parents.push_back(&result);
    other.parents.push_back(&result);
    return result;
}

// Display function to print data
void variable::print(bool matrix) const {
    if (matrix && dim2 > 1) { // Matrix format if matrix=true
        std::cout << "[";
        for (int i = 0; i < dim1; i++) {
            std::cout << "[";
            for (int j = 0; j < dim2; j++) {
                std::cout << this->data[i * dim2 + j];
                if (j < dim2 - 1) std::cout << ", ";
            }
            std::cout << "]";
            if (i < dim1 - 1) std::cout << ",\n ";
        }
        std::cout << "]" << std::endl;
    }
    else { // Original vector format if matrix=false or dim2 == 1
        std::cout << "[";
        for (int i = 0; i < dim1; i++) {
            std::cout << this->data[i];
            if (i < dim1 - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}
void variable::getChildren()
{
    for (auto i : this->children)
    {
        i->print();
    }
}

variable variable::dot( variable& other) 
{
    if (this->dim1 != other.dim1 || this->dim2 != other.dim2) {
        throw std::invalid_argument("Dimensions must match for dot product.");
    }
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    temp.push_back(const_cast<variable*>(&other));
    variable result(1, 1, false, temp);
    result.gradC11 = dim1;
    result.gradC12 = dim2;
    result.gradC21 = dim1;
    result.gradC22 = dim2;
    result.data = (float*)malloc(sizeof(float));
    result.gradientChild1 = (float*)malloc(dim1 * sizeof(float));
    result.gradientChild2 = (float*)malloc(dim1 * sizeof(float));
    cudaMemcpy(result.gradientChild1, other.data, dim1 * sizeof(float), cudaMemcpyHostToHost);
    cudaMemcpy(result.gradientChild2, this->data, dim1 * sizeof(float), cudaMemcpyHostToHost);
    *(result.data) = dotCUDA(this->data, other.data, dim1);
    this->parents.push_back(&result);
    return result;

}

variable variable::matrixMulVec( variable& other) 
{
    if (this->dim2 != other.dim1) {
        throw std::invalid_argument("Dimensions must match for mutliplication.");
    }
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    temp.push_back(const_cast<variable*>(&other));
    variable result(this->dim1, other.dim2, false, temp);
    result.data = (float*)malloc(this->dim1 * other.dim2 * sizeof(float));
    result.gradientChild1 = (float*)malloc(other.dim1* sizeof(float));
    result.gradientChild2 = (float*)malloc(this->dim1*this->dim2* sizeof(float));

    matrixVectorMul(this->data, other.data, result.data, this->dim1, this->dim2);

   
    std::memcpy(result.gradientChild1, other.data, other.dim1 * sizeof(float));
    result.gradC11 = other.dim2;
    result.gradC12 = other.dim1;

    // TRANSPOSE REMAINING
    transposeMatrixCPU(this->data, result.gradientChild2, dim1, dim2);
    result.gradC21 = this->dim2;
    result.gradC22 = this->dim1;
    this->parents.push_back(&result);
    other.parents.push_back(&result);
    return result;
}


void variable::tester()
{
    int dim1 = 3; // Number of rows
    int dim2 = 1; // Number of columns (vector)

    // True output vector
    float trueOutputData[3] = { 1.0f, 2.0f, 3.0f };

    // Predicted output vector
    float predOutputData[3] = { 1.5f, 2.5f, 3.5f };

    // Create variable objects for true and predicted outputs
    variable trueOutput(dim1, dim2, false, {});
    variable predOutput(dim1, dim2, false, {});

    // Set the data for true and predicted outputs
    trueOutput.setData(trueOutputData);
    predOutput.setData(predOutputData);

    // Calculate RMSE loss
    variable rmseResult = predOutput.RMSELOSS(&predOutput, &trueOutput);

    // Output RMSE value (result.data holds the RMSE)
    std::cout << "RMSE Loss: " << *(rmseResult.data) << std::endl;

    // Output RMSE gradient (result.gradientChild1 holds the gradient for predOutput)
    std::cout << "RMSE Gradient: ";
    for (int i = 0; i < dim1; i++) {
        std::cout << rmseResult.gradientChild1[i] << " ";
    }
    std::cout << std::endl;

    // Free allocated memory
    free(rmseResult.data);
    free(rmseResult.gradientChild1);

    
}



variable variable::sigmoid() const
{
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    variable result(this->dim1, this->dim2, false, temp);
    result.data = (float*)malloc(this->dim1 * this->dim2* sizeof(float));
    result.gradientChild1 = (float*)malloc(this->dim1 *this->dim2* sizeof(float));
    applySigmoid(this->data, result.data, dim1*dim2);

    sigmoidGradient(result.data, result.gradientChild1, dim1*dim2);
    return result;

}
variable variable::softmax() const
{
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    variable result(this->dim1, this->dim2, false, temp);
    result.data = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
    result.gradientChild1 = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
    applySoftmax(this->data, result.data, dim1* dim2);

    softmaxGradient(result.data, result.gradientChild1, dim1* dim2);
    return result;
}
variable variable::relu() const
{
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    variable result(this->dim1, this->dim2, false, temp);
    result.data = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
    result.gradientChild1 = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
    applyReLU(this->data, result.data, dim1 * dim2);

    reluGradient(result.data, result.gradientChild1, dim1 * dim2);
    return result;
}

void variable::backward(variable*x, float* gradAccum)
{
    if (this == x)
    {
        std::fill(backwardGrad, backwardGrad + dim1, 1.0f);
        for (auto i : this->children)
        {
            i->backward(x, gradAccum);
        }
    }
    else
    {
        if ((this->parents[0]->dim1 == 1) || (this->parents[0]->dim2 == 1))
        {
            if ((this->dim1 == 1) || (this->dim2 == 1))
            {

            }
        }
    }

}

variable variable::RMSELOSS(variable* output, variable* trueOutput)
{
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    variable result(this->dim1, this->dim2, false, temp);
    result.data = (float*)malloc(1 * sizeof(float));
    result.gradientChild1 = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
    result.gradC11 = dim1;
    result.gradC12 = dim2;

    *(result.data) = computeRMSE(output->data, trueOutput->data, dim1 * dim2);
    computeRMSEDerivative(output->data, trueOutput->data, result.gradientChild1, dim1, *(result.data));

    return result;


}

