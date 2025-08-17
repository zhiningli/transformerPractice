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


Eigen::MatrixXf LayerNorm::forward(const Eigen::MatrixXf& x){
    int seq_len = x.rows();

    last_input_ = x;

    last_mean_ = Eigen::VectorXf(seq_len);
    last_variance_ = Eigen::VectorXf(seq_len);

    last_normalized_ = Eigen::MatrixXf(seq_len, d_model_);

    for (int i = 0; i < seq_len; ++i){
        float mean = x.row(i).mean();
        last_mean_(i) = mean;

        Eigen::VectorXf centered = x.row(i).transpose().array() - mean;
        float variance = centered.array().square().mean();
        last_variance_(i) = variance;

        float std_dev = std::sqrt(variance + epsilon_);
        last_normalized_.row(i) = centered.array() / std_dev;
    }


    Eigen::MatrixXf output(seq_len, d_model_);
    
    for (int i = 0; i < seq_len; ++i) {
        // Apply gamma (scale) and beta (shift) element-wise to each row
        output.row(i) = (gamma_.array() * last_normalized_.row(i).array() + beta_.array()).matrix();
    }

    return output;
}


void LayerNorm::update_parameters(const Eigen::VectorXf& d_gamma, const Eigen::VectorXf& d_beta){
    gamma_ += d_gamma;
    beta_ +=  d_beta;
}

} //namespace transformer