#include "embedding.hpp"
#include <random>
#include <stdexcept>


namespace transformer {

TokenEmbedding::TokenEmbedding(int vocab_size, int embedding_dim): vocab_size_(vocab_size), embedding_dim_(embedding_dim){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(embedding_dim_));

    embedding_matrix_ = Eigen::MatrixXf(vocab_size_, embedding_dim_);

    for  (int i = 0; i < vocab_size_; ++i){
        for (int j = 0; j < embedding_dim_; ++j){
            embedding_matrix_(i, j) = dist(gen);
        }
    }
}

Eigen::MatrixXf TokenEmbedding::forward(const std::vector<int>& token_indices){
    int seq_len = static_cast<int>(token_indices.size());
    Eigen::MatrixXf output(seq_len, embedding_dim_);
    for (int i = 0; i < seq_len; ++i){
        int idx = token_indices[i];
        if (idx < 0 || idx >= vocab_size_){
            throw std::out_of_range("Token index out of vocabulary range");
        }
        output.row(i) = embedding_matrix_.row(idx);    
    }
    return output;
}


void TokenEmbedding::update_embedding_matrix(const Eigen::MatrixXf& gradients){

}


PositionalEncoding::PositionalEncoding(int max_seq_len, int embedding_dim): max_seq_len_(max_seq_len), embedding_dim_(embedding_dim){

    pos_encoding_ = Eigen::MatrixXf(max_seq_len_, embedding_dim_);

    for (int pos = 0; pos < max_seq_len_; ++pos){
        for (int i = 0;  i < embedding_dim_; i++){
            float denominator = std::pow(10000.0f, (2.0f * i / embedding_dim_));
            float angle = pos / denominator;
            if (i % 2 == 0){
                pos_encoding_(pos, i) = std::sin(angle);
            } else {
                pos_encoding_(pos, i) = std::cos(angle);
            }
        }
    }
}

Eigen::MatrixXf PositionalEncoding::forward(const Eigen::MatrixXf& token_embeddings){
    int seq_len = token_embeddings.rows();
    if (seq_len > max_seq_len_){
        throw std::out_of_range("Sequence length exceeds maximum sequence length");
    }
    return token_embeddings + pos_encoding_.block(0, 0, seq_len, embedding_dim_);
}



}