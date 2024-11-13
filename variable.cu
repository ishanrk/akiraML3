#include "variable.cuh"

variable::variable(int dimension1, int dimension2, bool random, std::vector<variable*>currChildren)
{
    dim1 = dimension1;
    dim2 = dimension2;
    rand = random;
    children = currChildren;
    opID = 0;
    if ((dim1 != 1) && (dim2 != 1))
    {
        matrix = true;
    }
    else
    {
        matrix = false;
    }
    data = (float*)malloc(dim1 * dim2 * sizeof(float));
    gradientChild1 = (float*)malloc(dim1 * dim2 * sizeof(float));
    if (currChildren.empty()) { children.push_back(this); }
    
    std::fill(gradientChild1, gradientChild1 + dim1, 1.0f);
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
    if (dim1 != 1 && dim2 != 1)
    {
        matrix = true;
    }
    else
    {
        matrix = false;
    }
    opID = 1;

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
    if (this->dim1 != 1 && this->dim2 != 1) {
        throw std::invalid_argument("Cannot be perfromed on matrix.");
    }
    opID = 1;
    matrix = false;
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
    this->parents.push_back(result);
    other.parents.push_back(result);
    return result;

}

variable variable::matrixMulVec( variable& other) 
{
    if (this->dim2 != other.dim1) {
        throw std::invalid_argument("Dimensions must match for mutliplication.");
    }
    matrix = false;
    opID = 2;
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    temp.push_back(const_cast<variable*>(&other));
    variable result(this->dim1, other.dim2, false, temp);
    result.data = (float*)malloc(this->dim1 * other.dim2 * sizeof(float));
    result.gradientChild1 = (float*)malloc(other.dim1* sizeof(float));
    result.gradientChild2 = (float*)malloc(this->dim1*this->dim2* sizeof(float));

    matrixVectorMul(this->data, other.data, result.data, this->dim1, this->dim2);

   
    std::memcpy(result.gradientChild1, other.data, other.dim1 * sizeof(float));
    
    std::memcpy(result.gradientChild2, this->data, this->dim1 * this->dim2* sizeof(float));
    // TRANSPOSE REMAINING

    this->parents.push_back(result);
    other.parents.push_back(result);
    return result;
}


void variable::tester()
{
    std::cout << "Test 1: 3x1 vector * 1x1 element\n";
    float A1[3] = { 1, 2, 3 };  // 3x1
    float B1[1] = { 5 };        // 1x1
    float C1[3];              // Result should be 3x1
    matrixMultiply(A1, B1, C1, 3, 1, 1);
    printMatrix(C1, 3, 1);  // Expected output: [5, 10, 15]

    // Test 2: 3x1 vector multiplied by a 1x3 vector (outer product)
    std::cout << "Test 2: 3x1 vector * 1x3 vector\n";
    float A2[3] = { 1, 2, 3 };      // 3x1
    float B2[3] = { 4, 5, 6 };      // 1x3
    float C2[9];                  // Result should be 3x3
    matrixMultiply(A2, B2, C2, 3, 1, 3);
    printMatrix(C2, 3, 3);  // Expected output: [4, 5, 6, 8, 10, 12, 12, 15, 18]

    // Test 3: 3x1 vector multiplied by a 1x1 element (alternative testing)
    std::cout << "Test 3: 3x1 vector * 1x1 element (same as Test 1)\n";
    float C3[3];              // Result should be 3x1
    matrixMultiply(A1, B1, C3, 3, 1, 1);
    printMatrix(C3, 3, 1);  // Expected output: [5, 10, 15]
    
}



variable variable::sigmoid() 
{
    if (dim1 != 1 && dim2 != 1)
    {
        matrix = true;
    }
    else
    {
        matrix = false;
    }
    opID = 1;
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    variable result(this->dim1, this->dim2, false, temp);
    result.data = (float*)malloc(this->dim1 * this->dim2* sizeof(float));
    result.gradientChild1 = (float*)malloc(this->dim1 *this->dim2* sizeof(float));
    applySigmoid(this->data, result.data, dim1*dim2);

    sigmoidGradient(result.data, result.gradientChild1, dim1*dim2);
    return result;

}
variable variable::softmax() 
{
    if (dim1 != 1 && dim2 != 1)
    {
        matrix = true;
    }
    else
    {
        matrix = false;
    }
    opID = 1;
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    variable result(this->dim1, this->dim2, false, temp);
    result.data = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
    result.gradientChild1 = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
    applySoftmax(this->data, result.data, dim1* dim2);

    softmaxGradient(result.data, result.gradientChild1, dim1* dim2);
    return result;
}
variable variable::relu() 
{
    if (dim1 != 1 && dim2 != 1)
    {
        matrix = true;
    }
    else
    {
        matrix = false;
    }
    opID = 1;
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
        std::fill(backwardGrad, backwardGrad + dim1, 1.0f);
       
        for (int x = 0; x<children.size();x++)
        {
 
            children[x]->backward(root,backwardGrad,x);
       
        }
    }
    else
    {
    
        backwardGrad = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
   
        if ((this->parents[0].dim1 == 1) && (this->parents[0].dim2 == 1))
        {
            // this indicates dot product 
           
       
            
            if ((this->dim1 == 1) && (this->dim2 == 1))
            {
             
                if (childID == 0)
                {
                    *(backwardGrad) = dotCUDA(gradAccum, parents[0].gradientChild1, dim1 * dim2);
                    
                }
                else
                {
                    *(backwardGrad) = dotCUDA(gradAccum, parents[0].gradientChild2, dim1 * dim2);
                   
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
               
                backwardGrad = parents[0].gradientChild1;
                
                for (int x = 0;x < children.size();x++)
                {

                    if (children[x] != this)
                    {

                        children[x]->backward(root, backwardGrad, x);
                    }

                }
            }


        }
        else if ((this->parents[0],dim1 == 1) || (this->parents[0].dim2 == 1))
        {
            if ((this->dim1 == 1) || (this->dim2 == 1))
            {
               
                if (childID== 0)
                {
                    
                    elementwiseMultiply(gradAccum, parents[0].gradientChild1, backwardGrad, dim1 * dim2);
                    
                }
                else
                {
                    
                    elementwiseMultiply(gradAccum, parents[0].gradientChild2, backwardGrad, dim1 * dim2);
                    
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
    matrix = false;
    opID = 1;
    if (this->dim1 != other.dim1 || this->dim2 != other.dim2) {
        throw std::invalid_argument("Dimensions must match for dot product.");
    }
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    temp.push_back(const_cast<variable*>(&other));
    variable result(dim1, dim2, false, temp);

    result.data = (float*)malloc(dim1*dim2*sizeof(float));
    result.gradientChild1 = (float*)malloc(dim1*dim2 * sizeof(float));
    result.gradientChild2 = (float*)malloc(dim1 *dim2* sizeof(float));
    cudaMemcpy(result.gradientChild1, other.data, dim1 * sizeof(float), cudaMemcpyHostToHost);
    cudaMemcpy(result.gradientChild2, this->data, dim1 * sizeof(float), cudaMemcpyHostToHost);
    elementwiseMultiply(this->data, other.data, result.data, dim1 * dim2);
    this->parents.push_back(result);
    other.parents.push_back(result);
    return result;
}

variable variable::RMSELOSS( variable &trueOutput)
{
    opID = 3;
    if (dim1 != 1 && dim2 != 1)
    {
        matrix = true;
    }
    else
    {
        matrix = false;
    }
    
    std::vector<variable*> temp;
    temp.push_back(const_cast<variable*>(this));
    variable result(1, 1, false, temp);
    result.data = (float*)malloc(1 * sizeof(float));
    result.gradientChild1 = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
  

    *(result.data) = computeRMSE(this->data, trueOutput.data, dim1 * dim2);
    computeRMSEDerivative(this->data, trueOutput.data, result.gradientChild1, dim1, *(result.data));
    
    this->parents.push_back(result);
    return result;


}

void variable::update(float lr)
{
    for (int x = 0; x < dim1 * dim2;x++)
    {
        std::cout << x << std::endl;
        data[x] = data[x] - lr * this->backwardGrad[x];
    }
    parents.pop_back();
}

void variable::reverseMode(float* gradAccum, int childID)
{
    if (parents.empty()) // root note
    {
        backwardGrad = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
        std::fill(backwardGrad, backwardGrad + dim1, 1.0f);
        if (opID != 0)
        {
            for (int x = 0; x < children.size();x++)
            {
                // lead node checker

                children[x]->reverseMode(backwardGrad, x);

            }
        }
    }
    else
    {
        backwardGrad = (float*)malloc(this->dim1 * this->dim2 * sizeof(float));
        if (opID == 3)
        {
            matrixMultiply(parents[0].gradientChild1, gradAccum, backwardGrad, dim1, 1, 1);
            
                for (int x = 0; x < children.size();x++)
                {
                    if (children[x] != this)
                    {
                        children[x]->reverseMode(backwardGrad, x);
                    }

                }
            
        }
        else if (opID == 2)
        {
            if (childID == 0)
            {
                if (matrix = true)
                {
                    matrixMultiply(gradAccum, parents[0].gradientChild1, backwardGrad, dim1, 1, dim2);
                }
                else
                {
                    matrixMultiply(gradAccum, parents[0].gradientChild1, backwardGrad, 1, dim1, dim2);
                }
            }
            else
            {
                if (matrix = true)
                {
                    matrixMultiply(gradAccum, parents[0].gradientChild2, backwardGrad, dim1, 1, dim2);
                }
                else
                {
                    matrixMultiply(gradAccum, parents[0].gradientChild2, backwardGrad, 1, dim1, dim2);
                }
            }

            
                for (int x = 0; x < children.size();x++)
                {
                    if (children[x] != this)
                    {
                        children[x]->reverseMode(backwardGrad, x);
                    }
                }
            
        }
        else if (opID == 1)
        {
            elementwiseMultiply(gradAccum, parents[0].gradientChild1, backwardGrad, dim1 * dim2);
            for (int x = 0; x < children.size();x++)
            {
                if (children[x] != this)
                {
                    children[x]->reverseMode(backwardGrad, x);
                }
            }
        }
        
    }
}


