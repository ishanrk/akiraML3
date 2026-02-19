#include "transformers.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
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
    W_ff1.push_back(variable(d_model, d_ff, false));
    b_ff1.push_back(variable(seq_len, d_ff, false));
    W_ff2.push_back(variable(d_ff, d_model, false));
    b_ff2.push_back(variable(seq_len, d_model, false));
    xavierInit2D(W_ff1[0].data, d_model, d_ff);
    xavierInit2D(W_ff2[0].data, d_ff, d_model);
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
    variable x_var(x.dim1, x.dim2, false);
    x_var.setData(x.data);
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
    variable ff_out = ff2_out + b_ff2[0];
    return residual1 + ff_out;
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

TransformerEncoder::TransformerEncoder(const TransformerEncoder& other)
    : max_len(other.max_len), d_model(other.d_model) {
    positional_encoding = (float*)malloc(max_len * d_model * sizeof(float));
    for (int i = 0; i < max_len * d_model; i++) positional_encoding[i] = other.positional_encoding[i];
    layers = other.layers;
    for (size_t i = 0; i < layers.size(); i++) {
        layers[i].optimizer = other.layers[i].optimizer;
    }
}

TransformerEncoder::~TransformerEncoder() {
    free(positional_encoding);
}

variable TransformerEncoder::forward(const variable& x) {
    variable x_pe(x.dim1, x.dim2, false);
    x_pe.setData(x.data);
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

static void writeMat(std::ofstream& f, const variable& v) {
    f << v.dim1 << " " << v.dim2 << "\n";
    int n = v.dim1 * v.dim2;
    for (int i = 0; i < n; i++) f << v.data[i] << "\n";
}
static void readMat(std::ifstream& f, variable& v) {
    int rows, cols;
    f >> rows >> cols;
    int n = rows * cols;
    for (int i = 0; i < n; i++) f >> v.data[i];
}

void TransformerEncoder::save(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("TransformerEncoder::save: cannot open " + path);
    f << "TRANSFORMER\nMAX_LEN\n" << max_len << "\nD_MODEL\n" << d_model << "\nNUM_HEADS\n"
      << layers[0].num_heads << "\nD_FF\n" << layers[0].d_ff << "\nNUM_LAYERS\n" << layers.size() << "\n";
    for (size_t L = 0; L < layers.size(); L++) {
        const auto& layer = layers[L];
        f << "LAYER\n" << L << "\n";
        f << "WQ\n"; writeMat(f, layer.W_q[0]);
        f << "WK\n"; writeMat(f, layer.W_k[0]);
        f << "WV\n"; writeMat(f, layer.W_v[0]);
        f << "WO\n"; writeMat(f, layer.W_o[0]);
        f << "WFF1\n"; writeMat(f, layer.W_ff1[0]);
        f << "BFF1\n"; writeMat(f, layer.b_ff1[0]);
        f << "WFF2\n"; writeMat(f, layer.W_ff2[0]);
        f << "BFF2\n"; writeMat(f, layer.b_ff2[0]);
    }
    f << "PE\n" << max_len << " " << d_model << "\n";
    for (int i = 0; i < max_len * d_model; i++) f << positional_encoding[i] << "\n";
    f << "END\n";
}

TransformerEncoder TransformerEncoder::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("TransformerEncoder::load: cannot open " + path);
    std::string tag;
    f >> tag;
    if (tag != "TRANSFORMER") throw std::runtime_error("TransformerEncoder::load: expected TRANSFORMER");
    int max_len_, d_model_, num_heads, d_ff, num_layers;
    f >> tag >> max_len_ >> tag >> d_model_ >> tag >> num_heads >> tag >> d_ff >> tag >> num_layers;
    TransformerEncoder enc(max_len_, d_model_, num_heads, d_ff, num_layers, nullptr);
    for (int L = 0; L < num_layers; L++) {
        f >> tag;
        if (tag != "LAYER") throw std::runtime_error("TransformerEncoder::load: expected LAYER");
        int idx;
        f >> idx;
        f >> tag; if (tag != "WQ") throw std::runtime_error("expected WQ"); readMat(f, enc.layers[L].W_q[0]);
        f >> tag; if (tag != "WK") throw std::runtime_error("expected WK"); readMat(f, enc.layers[L].W_k[0]);
        f >> tag; if (tag != "WV") throw std::runtime_error("expected WV"); readMat(f, enc.layers[L].W_v[0]);
        f >> tag; if (tag != "WO") throw std::runtime_error("expected WO"); readMat(f, enc.layers[L].W_o[0]);
        f >> tag; if (tag != "WFF1") throw std::runtime_error("expected WFF1"); readMat(f, enc.layers[L].W_ff1[0]);
        f >> tag; if (tag != "BFF1") throw std::runtime_error("expected BFF1"); readMat(f, enc.layers[L].b_ff1[0]);
        f >> tag; if (tag != "WFF2") throw std::runtime_error("expected WFF2"); readMat(f, enc.layers[L].W_ff2[0]);
        f >> tag; if (tag != "BFF2") throw std::runtime_error("expected BFF2"); readMat(f, enc.layers[L].b_ff2[0]);
    }
    f >> tag;
    if (tag != "PE") throw std::runtime_error("TransformerEncoder::load: expected PE");
    int pe_rows, pe_cols;
    f >> pe_rows >> pe_cols;
    for (int i = 0; i < max_len_ * d_model_; i++) f >> enc.positional_encoding[i];
    f >> tag;
    if (tag != "END") throw std::runtime_error("TransformerEncoder::load: expected END");
    return enc;
}
