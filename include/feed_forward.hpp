#pragma once

#include <Eigen/Dense>
#include <functional>

namespace transformer {

class FeedForward{
    private:
        int d_model_;
        int d_ff_;
    
        Eigen::MatrixXf W1_; // (d_model, d_ff)
        Eigen::MatrixXf W2_; // (d_ff, d_model_)

        Eigen::VectorXf b1_; // (d_ff)
        Eigen::VectorXf b2_; // (d_model_)

        Eigen::MatrixXf last_input_;
        Eigen::MatrixXf last_hidden_;

        /**
         * @brief Initialize weight matrices with Xavier initialization
         */
        void initialize_parameters();

        /**
         * @brief ReLU activation function
         * @param x Input matrix
         * @return ReLU(x) = max(0, x)
         */
        Eigen::MatrixXf relu(const Eigen::MatrixXf& x);

    public:
        FeedForward(int d_model, int d_ff);

        /**
         * @brief Forward pass of the feed-forward network
         * @param x Input matrix (seq_len, d_model)
         * @return Output matrix (seq_len, d_model)
         */
        Eigen::MatrixXf forward(const Eigen::MatrixXf& x);

        /**
         * @brief Update parameters (for training)
         * @param d_W1 Gradient for W1_
         * @param d_b1 Gradient for b1_
         * @param d_W2 Gradient for W2_
         * @param d_b2 Gradient for b2_
         */
        void update_parameters(const Eigen::MatrixXf& d_W1, const Eigen::VectorXf& d_b1,
        const Eigen::MatrixXf& d_W2, const Eigen::VectorXf& d_b2);

        
        /**
         * @brief Getter for testing
         */
        int get_d_model() const {return d_model_;}
        int get_d_ff() const {return d_ff_;}
        const Eigen::MatrixXf& get_W1() const {return W1_;}
        const Eigen::MatrixXf& get_W2() const {return W2_;}
        const Eigen::VectorXf& get_b1() const {return b1_;}
        const Eigen::VectorXf& get_b2() const {return b2_;}
        const Eigen::MatrixXf& get_last_input() const {return last_input_;}
        const Eigen::MatrixXf& get_last_hidden() const {return last_hidden_;}
};

} //namespace transformer