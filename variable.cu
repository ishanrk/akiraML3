#include "variable.cuh"

variable::variable(int dimension1, int dimension2, bool random, std::vector<variable*>currChildren)
{
    dim1 = dimension1;
    dim2 = dimension2;
    rand = random;
    children = currChildren;

    if (rand)
    {
        if (dim2 == 1)
        {
            data = (float*)malloc(dim1 * sizeof(float));
            gradient = (float*)malloc(dim1 * sizeof(float));
            
            random_init(data, dim1);
            // Set all values in `gradient` to 0
            std::fill(gradient, gradient + dim1, 0.0f);
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
    result.gradient = (float*)malloc(dim1 * sizeof(float));

    // Perform element-wise addition
    
    result.data = addWithCuda(result.data, this->data, other.data, dim1);
    
    // Set `gradient` of the new variable to 0
    std::fill(result.gradient, result.gradient + dim1, 0.0f);

    // Add both operands as children to the result

    return result;
}

// Display function to print data
void variable::print() const {
    
    std::cout << "[";
    for (int i = 0; i < dim1; i++) {
        std::cout << this->data[i];
        if (i < dim1 - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void variable::getChildren()
{
    for (auto i : this->children)
    {
        i->print();
    }
}
