#include "transformers.cuh"
#include <iostream>
#include <random>
#include <cmath>

variable scaledDotProductAttention(variable& Q, variable& K, variable& V) {
    variable Kt = K.transpose();
    variable scores = Q.matrixMul(Kt);
    float scale = 1.0f / sqrtf(static_cast<float>(Q.dim2));
    variable scaled = scores.scale(scale);
    variable attn_weights = scaled.rowSoftmax();
    variable output = attn_weights.matrixMul(V);
    return output;
}

void sinusoidalPositionalEncoding(float* pe, int max_len, int d_model) {
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d_model; i++) {
            float angle = static_cast<float>(pos) / powf(10000.0f, static_cast<float>(i) / d_model);
            if (i % 2 == 0) {
                pe[pos * d_model + i] = sinf(angle);
            } else {
                pe[pos * d_model + i] = cosf(angle);
            }
        }
    }
}

static void xavierInit2D(float* data, int rows, int cols) {
    float std = sqrtf(2.0f / (rows + cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std);
    for (int i = 0; i < rows * cols; i++) {
        data[i] = dist(gen);
    }
}

TransformerEncoderLayer::TransformerEncoderLayer(int seq_len_, int d_model_, int num_heads_, int d_ff_,
                                                  std::shared_ptr<Optimizer> opt)
    : seq_len(seq_len_), d_model(d_model_), num_heads(num_heads_), d_ff(d_ff_), optimizer(opt) {
    d_k = d_model / num_heads;
    if (d_k * num_heads != d_model) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
    W_q.push_back(variable(d_model, d_model, false));
    W_k.push_back(variable(d_model, d_model, false));
    W_v.push_back(variable(d_model, d_model, false));
    W_o.push_back(variable(d_model, d_model, false));
    xavierInit2D(W_q[0].data, d_model, d_model);
    xavierInit2D(W_k[0].data, d_model, d_model);
    xavierInit2D(W_v[0].data, d_model, d_model);
    xavierInit2D(W_o[0].data, d_model, d_model);
    W_ff1.push_back(variable(d_ff, d_model, false));
    b_ff1.push_back(variable(seq_len, d_ff, false));
    W_ff2.push_back(variable(d_model, d_ff, false));
    b_ff2.push_back(variable(seq_len, d_model, false));
    xavierInit2D(W_ff1[0].data, d_ff, d_model);
    xavierInit2D(W_ff2[0].data, d_model, d_ff);
    for (int i = 0; i < seq_len * d_ff; i++) b_ff1[0].data[i] = 0.1f;
    for (int i = 0; i < seq_len * d_model; i++) b_ff2[0].data[i] = 0.1f;
    if (optimizer) {
        W_q[0].setOptimizer(optimizer);
        W_k[0].setOptimizer(optimizer);
        W_v[0].setOptimizer(optimizer);
        W_o[0].setOptimizer(optimizer);
        W_ff1[0].setOptimizer(optimizer);
        b_ff1[0].setOptimizer(optimizer);
        W_ff2[0].setOptimizer(optimizer);
        b_ff2[0].setOptimizer(optimizer);
    }
}

variable TransformerEncoderLayer::forward(const variable& x) {
    variable x_var = x;
    variable Q = x_var.matrixMul(W_q[0]);
    variable K = x_var.matrixMul(W_k[0]);
    variable V = x_var.matrixMul(W_v[0]);
    variable attn_out = scaledDotProductAttention(Q, K, V);
    variable proj = attn_out.matrixMul(W_o[0]);
    variable residual1 = x_var + proj;
    variable ff1_out = residual1.matrixMul(W_ff1[0]);
    variable ff1_bias = ff1_out + b_ff1[0];
    variable ff1_relu = ff1_bias.relu();
    variable ff2_out = ff1_relu.matrixMul(W_ff2[0]);
    variable out = ff2_out + b_ff2[0];
    return residual1 + out;
}

TransformerEncoder::TransformerEncoder(int max_len_, int d_model_, int num_heads, int d_ff, int num_layers,
                                       std::shared_ptr<Optimizer> opt)
    : max_len(max_len_), d_model(d_model_) {
    positional_encoding = (float*)malloc(max_len * d_model * sizeof(float));
    sinusoidalPositionalEncoding(positional_encoding, max_len, d_model);
    for (int i = 0; i < num_layers; i++) {
        layers.push_back(TransformerEncoderLayer(max_len, d_model, num_heads, d_ff, opt));
    }
}

TransformerEncoder::~TransformerEncoder() {
    free(positional_encoding);
}

variable TransformerEncoder::forward(const variable& x) {
    variable x_pe = x;
    for (int i = 0; i < x.dim1 * x.dim2; i++) {
        x_pe.data[i] += positional_encoding[i];
    }
    variable out = x_pe;
    for (size_t i = 0; i < layers.size(); i++) {
        out = layers[i].forward(out);
    }
    return out;
}

void TransformerEncoder::setOptimizer(std::shared_ptr<Optimizer> opt) {
    for (size_t i = 0; i < layers.size(); i++) {
        layers[i].optimizer = opt;
        layers[i].W_q[0].setOptimizer(opt);
        layers[i].W_k[0].setOptimizer(opt);
        layers[i].W_v[0].setOptimizer(opt);
        layers[i].W_o[0].setOptimizer(opt);
        layers[i].W_ff1[0].setOptimizer(opt);
        layers[i].b_ff1[0].setOptimizer(opt);
        layers[i].W_ff2[0].setOptimizer(opt);
        layers[i].b_ff2[0].setOptimizer(opt);
    }
}
