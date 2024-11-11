#include "variable.cuh"

variable::variable(int dimension1, int dimension2, bool random, std::vector<variable*>currChildren)
{
    dim1 = dimension1;
    dim2 = dimension2;
    rand = random;
    children = currChildren;
    
    if (currChildren.empty()) { children.push_back(this); }

    if (rand)
    {
        
            data = (float*)malloc(dim1 *dim2* sizeof(float));
            gradientChild1 = (float*)malloc(dim1 *dim2* sizeof(float));
            random_init(data, dim1, dim2);
            // Set all values in `gradient` to 0
            std::fill(gradientChild1, gradientChild1 + dim1, 1.0f);
        
    }
}

int variable::setData(float* arr)
{

        for (int x = 0; x < this->dim1*this->dim2;x++)
        {
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
    result.gradientChild1 = (float*)malloc(this->dim1 * this->dim2* sizeof(float));
    result.gradientChild2 = (float*)malloc(other.dim1 * other.dim2* sizeof(float));

    matrixVectorMul(this->data, other.data, result.data, this->dim1, this->dim2);

    for (int i = 0; i < this->dim1; i++) {
        std::memcpy(result.gradientChild1 + i * other.dim1, other.data, other.dim1 * sizeof(float));
    }

    float* rowVec = new float[this->dim1];
    std::fill(rowVec, rowVec + this->dim1, 1.0f);
    rowMatrixMul(rowVec, this->data, gradientChild2, this->dim1, other.dim2);
    this->parents.push_back(&result);
    other.parents.push_back(&result);
    return result;
}


void variable::tester()
{
    int n = 2; // Row vector size
    int m = 3; // Matrix column size

    // Example row and matrix
    float row[2] = { 1.0f, 1.0f};
    float matrix[6] = { 1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f};
    float result[3]; // Result vector of size 4

    // Perform row * matrix multiplication
    rowMatrixMul(row, matrix, result, n, m);

    // Print the result
    std::cout << "Result: ";
    for (int i = 0; i < m; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
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