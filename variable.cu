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
        if (dim2 == 1)
        {
            data = (float*)malloc(dim1 * sizeof(float));
            gradientChild1 = (float*)malloc(dim1 * sizeof(float));
            
            random_init(data, dim1);
            // Set all values in `gradient` to 0
            std::fill(gradientChild1, gradientChild1 + dim1, 0.0f);
        }
    }
}

int variable::setData(float* arr, int dimension1)
{
    if (dimension1 != dim1)
    {
        
    }
    else
    {
        for (int x = 0; x < dimension1;x++)
        {
            this->data[x] = arr[x];
        }
    }
    return 1;
}

variable::~variable()
{
  
}
variable variable::operator+(const variable& other) const {
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

variable variable::dot(const variable& other) const
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

    return result;

}

variable variable::matrixMulVec(const variable& other) const
{
    if (this->dim2 != other.dim1) {
        throw std::invalid_argument("Dimensions must match for mutliplication.");
    }
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    temp.push_back(const_cast<variable*>(&other));
    variable result(this->dim1, other.dim2, false, temp);
    result.data = (float*)malloc(this->dim1 * other.dim2 * sizeof(float));
    result.gradientChild1 = (float*)malloc(other.dim1 * sizeof(float));
    result.gradientChild2 = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));

    matrixVectorMul(this->data, other.data, result.data, this->dim1, this->dim2);

    std::memcpy(result.gradientChild1, other.data, other.dim1 * sizeof(float));

    // Fill gradientChild2 with 'this->data'
    std::memcpy(result.gradientChild2, this->data, this->dim1 * this->dim2 * sizeof(float));

    return result;
}


void variable::tester()
{
    int M = 4; // Number of rows in A
    int N = 3; // Number of columns in A, and size of x

    float A[4][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12}
    };

    float x[3] = { 1, 2, 3 }; // Vector x
    float y[4]; // Result vector y

    // Perform matrix-vector multiplication
    matrixVectorMul(&A[0][0], x, y, M, N);

    // Print the result
    std::cout << "Result y = A * x:" << std::endl;
    for (int i = 0; i < M; i++) {
        std::cout << y[i] << std::endl;
    }

}