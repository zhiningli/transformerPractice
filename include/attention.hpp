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
         * @brief Compute the sacled dot-product attention
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

} // namespace transformer