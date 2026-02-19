#pragma once
#include "variable.cuh"
#include "kernel.cuh"
#include <cstring>

// Base optimizer class
class Optimizer {
public:
    virtual void update(variable& param, float* gradients, int iteration) = 0;
    virtual ~Optimizer() = default;
};

// Adam optimizer class
class Adam : public Optimizer {
private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float* m;  // First moment estimate
    float* v;  // Second moment estimate
    int param_size;
    int iteration_count;

public:
    Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f) 
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), m(nullptr), v(nullptr), param_size(0), iteration_count(0) {}
    
    ~Adam() {
        if (m) free(m);
        if (v) free(v);
    }
    
    void initialize(int size) {
        param_size = size;
        m = (float*)malloc(static_cast<size_t>(size) * sizeof(float));
        v = (float*)malloc(static_cast<size_t>(size) * sizeof(float));
        std::memset(m, 0, static_cast<size_t>(size) * sizeof(float));
        std::memset(v, 0, static_cast<size_t>(size) * sizeof(float));
    }
    
    void update(variable& param, float* gradients, int iteration) override {
        if (m == nullptr || v == nullptr) {
            initialize(param.dim1 * param.dim2);
        }
        
        iteration_count = iteration + 1;
        adamUpdate(param.data, gradients, m, v, learning_rate, beta1, beta2, epsilon, iteration_count, param_size);
    }
};

// RMSprop optimizer class
class RMSprop : public Optimizer {
private:
    float learning_rate;
    float decay_rate;
    float epsilon;
    float* v;  // Running average of squared gradients
    int param_size;

public:
    RMSprop(float lr = 0.001f, float decay = 0.9f, float eps = 1e-8f) 
        : learning_rate(lr), decay_rate(decay), epsilon(eps), v(nullptr), param_size(0) {}
    
    ~RMSprop() {
        if (v) free(v);
    }
    
    void initialize(int size) {
        param_size = size;
        v = (float*)malloc(static_cast<size_t>(size) * sizeof(float));
        std::memset(v, 0, static_cast<size_t>(size) * sizeof(float));
    }
    
    void update(variable& param, float* gradients, int iteration) override {
        if (v == nullptr) {
            initialize(param.dim1 * param.dim2);
        }
        
        rmspropUpdate(param.data, gradients, v, learning_rate, decay_rate, epsilon, param_size);
    }
};
