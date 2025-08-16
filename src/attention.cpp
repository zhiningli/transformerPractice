#include "attention.hpp"
#include <iostream>
#include <random>
#include <stdexcept>

namespace transformer {

ScaledDotProductAttention::ScaledDotProductAttention(int d_k): scale_factor_(1.0f / std::sqrt(d_k)){
}


Eigen::MatrixXf ScaledDotProductAttention::softmax(const Eigen::MatrixXf& input){
    Eigen::MatrixXf result = input;

    for (int i = 0; i < result.rows(); ++i){
        float max_val = result.row(i).maxCoeff();

        for (int j = 0; j < result.cols(); ++j){
            result(i, j) = std::exp(result(i, j) - max_val);
        }

        float sum = result.row(i).sum();

        result.row(i) /= sum;
    }
    return result;
}


Eigen::MatrixXf ScaledDotProductAttention::forward(
    const Eigen::MatrixXf& Q,
    const Eigen::MatrixXf& K,
    const Eigen::MatrixXf& V 
){
    Eigen::MatrixXf scores = Q * K.transpose();
    scores *= scale_factor_;
    Eigen::MatrixXf attention_weights = softmax(scores);

    Eigen::MatrixXf output = attention_weights * V;
    return output;
}


MultiHeadAttention::MultiHeadAttention(int num_heads, int d_model): num_heads_(num_heads), d_model_(d_model), attention_(d_model / num_heads){
    if (d_model % num_heads != 0){
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }

    d_k_ = d_model / num_heads;
    d_v_ = d_model / num_heads;

    initialize_weights();
}


void MultiHeadAttention::initialize_weights(){
    float limit = std::sqrt(6.0f / (d_model_ + d_k_));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    W_q_ = Eigen::MatrixXf(d_model_, d_model_);
    W_k_ = Eigen::MatrixXf(d_model_, d_model_);
    W_v_ = Eigen::MatrixXf(d_model_, d_model_);
    W_o_ = Eigen::MatrixXf(d_model_, d_model_);

    for (int i = 0; i < d_model_; ++i){
        for (int j = 0; j < d_model_; ++j){
            W_q_(i, j) = dist(gen);
            W_k_(i, j) = dist(gen);
            W_v_(i, j) = dist(gen);
            W_o_(i, j) = dist(gen);
        }
    }

    b_q_ = Eigen::VectorXf::Zero(d_model_);
    b_k_ = Eigen::VectorXf::Zero(d_model_);
    b_v_ = Eigen::VectorXf::Zero(d_model_);
    b_o_ = Eigen::VectorXf::Zero(d_model_);
}


Eigen::MatrixXf MultiHeadAttention::reshape_for_heads(const Eigen::MatrixXf& input){
    int seq_len = input.rows();

    Eigen::MatrixXf reshaped(seq_len * num_heads_, d_k_);

    for (int seq = 0; seq < seq_len; ++seq){
        for (int head = 0; head < num_heads_; ++head){
            int output_row = seq * num_heads_ + head;
            int input_start_col = head * d_k_;

            reshaped.row(output_row) = input.block(seq, input_start_col, 1, d_k_);
        }
    }
    return reshaped;

}


Eigen::MatrixXf MultiHeadAttention::reshape_from_heads(const Eigen::MatrixXf& input){
    Eigen::MatrixXf reshaped(input.rows() / num_heads_, d_model_);

    for (int seq = 0; seq < input.rows() / num_heads_; ++seq){
        for (int head = 0; head < num_heads_; ++head){
            int input_row = seq * num_heads_ + head;
            int output_start_col = head * d_v_;

            reshaped.block(seq, output_start_col, 1, d_v_) = input.row(input_row);
        }
    }
    return reshaped;
}


Eigen::MatrixXf MultiHeadAttention::forward(const Eigen::MatrixXf& query, 
                                            const Eigen::MatrixXf& key, 
                                            const Eigen::MatrixXf& value,
                                            const Eigen::MatrixXf& mask){
    int seq_len = query.rows();                                       
    
    //Linear projection for all heads
    Eigen::MatrixXf Q = query * W_q_.transpose() + b_q_.transpose().replicate(seq_len, 1);
    Eigen::MatrixXf K = key * W_k_.transpose() + b_k_.transpose().replicate(seq_len, 1);
    Eigen::MatrixXf V = value * W_v_.transpose() + b_v_.transpose().replicate(seq_len, 1);

    //Reshape for multihead processing
    Eigen::MatrixXf Q_heads = reshape_for_heads(Q);
    Eigen::MatrixXf K_heads = reshape_for_heads(K);
    Eigen::MatrixXf V_heads = reshape_for_heads(V);

    Eigen::MatrixXf attention_output = attention_.forward(Q_heads, K_heads, V_heads);

    Eigen::MatrixXf concatenated = reshape_from_heads(attention_output);
    Eigen::MatrixXf output = concatenated * W_o_.transpose() + b_o_.transpose().replicate(seq_len, 1);

    return output;
    }

}