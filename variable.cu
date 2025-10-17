#include "variable.cuh"
#include "optimizers.cuh"

variable::variable(int dimension1, int dimension2, bool random, std::vector<variable*>currChildren)
{
    dim1 = dimension1;
    dim2 = dimension2;
    rand = random;
    children = currChildren;
    data = (float*)malloc(dim1 * dim2 * sizeof(float));
    gradientChild1 = (float*)malloc(dim1 * dim2 * sizeof(float));
    optimizer = nullptr;
    if (currChildren.empty()) { children.push_back(this); }
    gradC11 = dim1;
    gradC12 = dim2;
    std::fill(gradientChild1, gradientChild1 + dim1 * dim2, 1.0f);

    if (rand)
    { 
            random_init(data, dim1, dim2);
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

    // Allocate memory for `result` data and gradient
    result.data = (float*)malloc(dim1 * dim2 * sizeof(float));
    result.gradientChild1 = (float*)malloc(dim1 * dim2 * sizeof(float));
    result.gradientChild2 = (float*)malloc(dim1 * dim2 * sizeof(float));
    result.gradC11 = dim1;
    result.gradC12 = dim2;
    result.gradC21 = dim1;
    result.gradC22 = dim2;
    // Perform element-wise addition
    
    result.data = addWithCuda(result.data, this->data, other.data, dim1);
  
    // Set `gradient` of the new variable to 1
    std::fill(result.gradientChild1, result.gradientChild1 + dim1 * dim2, 1.0f);
    std::fill(result.gradientChild2, result.gradientChild2 + dim1 * dim2, 1.0f);

    // Add both operands as children to the result
    this->parents.push_back(result);
    other.parents.push_back(result);
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
    result.gradientChild1 = (float*)malloc(dim1 * dim2 * sizeof(float));
    result.gradientChild2 = (float*)malloc(dim1 * dim2 * sizeof(float));
    cudaMemcpy(result.gradientChild1, other.data, dim1 * dim2 * sizeof(float), cudaMemcpyHostToHost);
    cudaMemcpy(result.gradientChild2, this->data, dim1 * dim2 * sizeof(float), cudaMemcpyHostToHost);
    *(result.data) = dotCUDA(this->data, other.data, dim1);
    this->parents.push_back(result);
    other.parents.push_back(result);
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
    result.gradientChild1 = (float*)malloc(other.dim1 * other.dim2 * sizeof(float));
    result.gradientChild2 = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));

    matrixVectorMul(this->data, other.data, result.data, this->dim1, this->dim2);

   
    std::memcpy(result.gradientChild1, other.data, other.dim1 * other.dim2 * sizeof(float));
    result.gradC11 = other.dim2;
    result.gradC12 = other.dim1;

    // TRANSPOSE REMAINING
    transposeMatrixCPU(this->data, result.gradientChild2, dim1, dim2);
    result.gradC21 = this->dim2;
    result.gradC22 = this->dim1;
    this->parents.push_back(result);
    other.parents.push_back(result);
    return result;
}


void variable::tester()
{
    variable var1(3, 1, false, {}); // Creates a 3x1 variable
    variable var2(3, 1, false, {}); // Creates a 3x1 variable

    // Allocate data for the two variables (manually filling them for simplicity)
    float data1[] = { 1.0f, 2.0f, 3.0f }; // Set data for var1
    float data2[] = { 4.0f, 5.0f, 6.0f }; // Set data for var2

    // Set data in each variable
    var1.setData(data1);
    var2.setData(data2);

    // Perform element-wise multiplication
    variable result = var1.elementWise(var2);

    // Print result data
    std::cout << "Element-wise multiplication result: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << result.data[i] << " ";
    }
    std::cout << std::endl;

    // Print gradients (gradients should just be copies of the other vector in this case)
    std::cout << "Gradient w.r.t. var1 (gradientChild1): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << result.gradientChild1[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Gradient w.r.t. var2 (gradientChild2): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << result.gradientChild2[i] << " ";
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

int variable::backward(variable * root, float* gradAccum, int childID)
{
    
    if (this == root)
    {
    
        backwardGrad = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
        std::fill(backwardGrad, backwardGrad + dim1 * dim2, 1.0f);
       
        for (int x = 0; x<children.size();x++)
        {
 
            children[x]->backward(root,backwardGrad,x);
       
        }
    }
    else
    {
    
        backwardGrad = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
   
        if (!parents.empty() && (this->parents[0].dim1 == 1) && (this->parents[0].dim2 == 1))
        {
            // this indicates dot product 
           
       
            
            if ((this->dim1 == 1) && (this->dim2 == 1))
            {
             
                if (!parents.empty()) {
                    if (childID == 0 && parents[0].gradientChild1 != nullptr)
                    {
                        *(backwardGrad) = dotCUDA(gradAccum, parents[0].gradientChild1, dim1 * dim2);
                        
                    }
                    else if (childID == 1 && parents[0].gradientChild2 != nullptr)
                    {
                        *(backwardGrad) = dotCUDA(gradAccum, parents[0].gradientChild2, dim1 * dim2);
                       
                    }
                }
                
                
                for (int x = 0;x < children.size();x++)
                {
                    
                    if (children[x] != this)
                    {
                        
                        children[x]->backward(root, backwardGrad, x);
                    }
                    
                }
            }
            else
            {
               
                if (!parents.empty() && parents[0].gradientChild1 != nullptr) {
                    backwardGrad = parents[0].gradientChild1;
                }
                
                for (int x = 0;x < children.size();x++)
                {

                    if (children[x] != this)
                    {

                        children[x]->backward(root, backwardGrad, x);
                    }

                }
            }


        }
        else if (!parents.empty() && ((this->parents[0].dim1 == 1) || (this->parents[0].dim2 == 1)))
        {
            if ((this->dim1 == 1) || (this->dim2 == 1))
            {
               
                if (!parents.empty()) {
                    if (childID == 0 && parents[0].gradientChild1 != nullptr)
                    {
                        elementwiseMultiply(gradAccum, parents[0].gradientChild1, backwardGrad, dim1 * dim2);
                    }
                    else if (childID == 1 && parents[0].gradientChild2 != nullptr)
                    {
                        elementwiseMultiply(gradAccum, parents[0].gradientChild2, backwardGrad, dim1 * dim2);
                    }
                }
                
                for (int x = 0; x<children.size(); x++)
                {
                   
                    if (children[x] != this)
                    {
                       
                        children[x]->backward(root, backwardGrad, x);
                     
                    }
                }
            }
        }
        
    }
    return 1;

}

variable variable::elementWise(variable& other)
{
    if (this->dim1 != other.dim1 || this->dim2 != other.dim2) {
        throw std::invalid_argument("Dimensions must match for dot product.");
    }
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    temp.push_back(const_cast<variable*>(&other));
    variable result(dim1, dim2, false, temp);
    result.gradC11 = dim1;
    result.gradC12 = dim2;
    result.gradC21 = dim1;
    result.gradC22 = dim2;
    result.data = (float*)malloc(dim1*dim2*sizeof(float));
    result.gradientChild1 = (float*)malloc(dim1*dim2 * sizeof(float));
    result.gradientChild2 = (float*)malloc(dim1 *dim2* sizeof(float));
    cudaMemcpy(result.gradientChild1, other.data, dim1 * dim2 * sizeof(float), cudaMemcpyHostToHost);
    cudaMemcpy(result.gradientChild2, this->data, dim1 * dim2 * sizeof(float), cudaMemcpyHostToHost);
    elementwiseMultiply(this->data, other.data, result.data, dim1 * dim2);
    this->parents.push_back(result);
    other.parents.push_back(result);
    return result;
}

variable variable::RMSELOSS( variable &trueOutput)
{
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    variable result(1, 1, false, temp);
    result.data = (float*)malloc(1 * sizeof(float));
    result.gradientChild1 = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
    result.gradC11 = dim1;
    result.gradC12 = dim2;

    *(result.data) = computeRMSE(this->data, trueOutput.data, dim1 * dim2);
    computeRMSEDerivative(this->data, trueOutput.data, result.gradientChild1, dim1, *(result.data));
    
    this->parents.push_back(result);
    return result;


}

void variable::update(float lr)
{
    if (backwardGrad != nullptr) {
        for (int x = 0; x < dim1 * dim2; x++)
        {
            data[x] = data[x] - lr * this->backwardGrad[x];
        }
    }
    if (!parents.empty()) {
        parents.pop_back();
    }
}

void variable::setOptimizer(std::shared_ptr<Optimizer> opt)
{
    optimizer = opt;
}

void variable::updateWithOptimizer(int iteration)
{
    if (optimizer != nullptr && backwardGrad != nullptr)
    {
        optimizer->update(*this, backwardGrad, iteration);
    }
    else
    {
        std::cout << "Warning: No optimizer set or no gradients available" << std::endl;
    }
}


