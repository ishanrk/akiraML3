#pragma once
#include "variable.cuh"
#include "kernel.cuh"
#include "optimizers.cuh"
#include <vector>
#include <memory>
#include <cmath>

// Scaled dot-product attention: output = rowSoftmax(Q * K^T / sqrt(d_k)) * V
// Q (seq_len, d_k), K (seq_len, d_k), V (seq_len, d_v) -> (seq_len, d_v)
variable scaledDotProductAttention(variable& Q, variable& K, variable& V);

// Sinusoidal positional encoding: fill pe[0..seq_len-1][0..d_model-1]
void sinusoidalPositionalEncoding(float* pe, int max_len, int d_model);

// Transformer encoder layer: multi-head self-attention + residual + feed-forward + residual
class TransformerEncoderLayer {
public:
    int seq_len;
    int d_model;
    int num_heads;
    int d_k;           // d_model / num_heads
    int d_ff;          // feed-forward hidden dim
    std::vector<variable> W_q, W_k, W_v, W_o;
    std::vector<variable> W_ff1, b_ff1, W_ff2, b_ff2;
    std::shared_ptr<Optimizer> optimizer;

    TransformerEncoderLayer(int seq_len, int d_model, int num_heads, int d_ff,
                            std::shared_ptr<Optimizer> opt = nullptr);
    variable forward(const variable& x);
};

// Stack of encoder layers
class TransformerEncoder {
public:
    std::vector<TransformerEncoderLayer> layers;
    int max_len;
    int d_model;
    float* positional_encoding;

    TransformerEncoder(int max_len, int d_model, int num_heads, int d_ff, int num_layers,
                      std::shared_ptr<Optimizer> opt = nullptr);
    TransformerEncoder(const TransformerEncoder& other);
    TransformerEncoder& operator=(const TransformerEncoder&) = delete;
    ~TransformerEncoder();
    variable forward(const variable& x);
    void setOptimizer(std::shared_ptr<Optimizer> opt);
    void save(const std::string& path) const;
    static TransformerEncoder load(const std::string& path);
};
