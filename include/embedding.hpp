#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>

namespace transformer {


class TokenEmbedding {
    private:
        Eigen::MatrixXf embedding_matrix_;
        int vocab_size_;
        int embedding_dim_;

    public:
        TokenEmbedding(int vocab_size, int embedding_dim);

        Eigen::MatrixXf forward(const std::vector<int>& token_indices);

        const Eigen::MatrixXf& get_embedding_matrix() const {return embedding_matrix_;}

        void update_embedding_matrix(const Eigen::MatrixXf& gradients);

};


class PositionalEncoding {
    private:
        Eigen::MatrixXf pos_encoding_;
        int max_seq_len_;
        int embedding_dim_;

    public: 
        PositionalEncoding(int max_seq_len, int embedding_dim);
        Eigen::MatrixXf forward(const Eigen::MatrixXf& token_embeddings);
        const Eigen::MatrixXf& get_pos_encoding() const {return pos_encoding_;};
};

}