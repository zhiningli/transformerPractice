#include "layer_norm.hpp"
#include <cmath>
#include <Eigen/Dense>

namespace transformer {

LayerNorm::LayerNorm(int d_model, float epsilon) : d_model_(d_model), epsilon_(epsilon) {
    initialize_parameters();
}

void LayerNorm::initialize_parameters(){
    gamma_ = Eigen::VectorXf::Ones(d_model_);
    beta_ = Eigen::VectorXf::Zero(d_model_);

}


Eigen::MatrixXf LayerNorm::forward(const Eigen::MatrixXf& x) {
    int seq_len = x.rows();

    last_input_ = x;
    last_mean_ = Eigen::VectorXf(seq_len);
    last_variance_ = Eigen::VectorXf(seq_len);
    last_normalized_ = Eigen::MatrixXf(seq_len, d_model_);

    for (int i = 0; i < seq_len; ++i) {
        // Calculate mean
        float mean = x.row(i).mean();
        last_mean_(i) = mean;

        // Calculate variance
        Eigen::RowVectorXf row = x.row(i);
        float variance = (row.array() - mean).square().mean();
        last_variance_(i) = variance;

        // Normalize: (x - mean) / sqrt(variance + epsilon)
        float std_dev = std::sqrt(variance + epsilon_);
        last_normalized_.row(i) = (row.array() - mean) / std_dev;
    }

    // Apply gamma and beta element-wise along the feature dimension
    Eigen::MatrixXf output(seq_len, d_model_);
    for (int i = 0; i < seq_len; ++i) {
        output.row(i) = last_normalized_.row(i).cwiseProduct(gamma_.transpose()) + beta_.transpose();
    }

    return output;
}


void LayerNorm::update_parameters(const Eigen::VectorXf& d_gamma, const Eigen::VectorXf& d_beta){
    gamma_ += d_gamma;
    beta_ +=  d_beta;
}

} //namespace transformer