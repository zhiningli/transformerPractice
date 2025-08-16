#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace transformer {

class ScaledDotProductAttention{
    private:
        float scale_factor_;

        Eigen::MatrixXf softmax(const Eigen::MatrixXf& input);
        
    public:
        /**
         * @brief Constructor
         * @param d_k: Dimension of the key
         */
        ScaledDotProductAttention(int d_k);

        /**
         * @brief Compute the scaled dot-product attention
         * @param Q: Query matrix of shape (seq_len, d_k)
         * @param K: Key matrix of shape (seq_len, d_k)
         * @param V: Value matrix of shape (seq_len, d_v)
         * @return Attention output of shape (seq_len, d_V)
         */
        Eigen::MatrixXf forward(const Eigen::MatrixXf& Q,
                                const Eigen::MatrixXf& K,
                                const Eigen::MatrixXf& V);

        /**
         * @brief Get the scale factor
         */
        float get_scale_factor() const {return scale_factor_;};
};



class MultiHeadAttention{
    private:
        int num_heads_;
        int d_model_;
        int d_k_;
        int d_v_;

        Eigen::MatrixXf W_q_; // (d_model, d_model)
        Eigen::MatrixXf W_k_; // (d_model, d_model)
        Eigen::MatrixXf W_v_; // (d_model, d_model)
        Eigen::MatrixXf W_o_; // (d_model, d_model)

        Eigen::VectorXf b_q_; // (d_model)
        Eigen::VectorXf b_k_; // (d_model)
        Eigen::VectorXf b_v_; // (d_model)
        Eigen::VectorXf b_o_; // (d_model)

        ScaledDotProductAttention attention_;

    public:
        MultiHeadAttention(int num_heads, int d_model);

        /**
         * @brief Forward pass of multi-head attention
         * @param query: Query matrix of shape (seq_len, d_model)
         * @param key: Key matrix of shape (seq_len, d_model)
         * @param value: Value matrix of shape (seq_len, d_model)
         * @param mask: Optional attention mask of shape (seq_len, seq_len)
         * @return Attention output of shape (seq_len, d_model)
         */
        Eigen::MatrixXf forward(const Eigen::MatrixXf& query,
                                const Eigen::MatrixXf& key,
                                const Eigen::MatrixXf& value,
                                const Eigen::MatrixXf& mask = Eigen::MatrixXf());

        /**
         * @brief Initialize weights with Xavior/Glorot initialization
         */
        void initialize_weights();

        /**
         * @brief Get the scale factor used by attention head
         */
        float get_scale_factor() const {return attention_.get_scale_factor();};
    
    private:
        /**
         * @brief Reshape maxtrix for multi-head processing
         * @param input matrix of shape (seq_len, d_model)
         * @return Reshaped matrix of shape (seq_len * num_heads, d_k)
         */
        Eigen::MatrixXf reshape_for_heads(const Eigen::MatrixXf& input);

        /**
         * @brief Reshape back from multi-head format to original shape
         * @param input matrix of shape (seq_len * num_heads, d_k)
         * @return Reshaped matrix of shape (seq_len, d_model)
         */
        Eigen::MatrixXf reshape_from_heads(const Eigen::MatrixXf& input);


};


} // namespace transformer