#include "feed_forward.hpp"
#include <random>
#include <cmath>
#include <algorithm>

namespace transformer {

FeedForward::FeedForward(int d_model, int d_ff) : d_model_(d_model), d_ff_(d_ff){
    initialize_parameters();
}


void FeedForward::initialize_parameters(){
    std::random_device rd;
    std::mt19937 gen(rd());
    
    float limit1 = std::sqrt(6.0f / (d_model_ + d_ff_));
    std::uniform_real_distribution<float> dist1(-limit1, limit1);

    W1_ = Eigen::MatrixXf(d_model_, d_ff_);

    for (int i = 0; i < d_model_; ++i){
        for (int j = 0; j < d_ff_; ++j){
            W1_(i, j) = dist1(gen);
        }
    }

    float limit2 = std::sqrt(6.0f / (d_ff_ + d_model_));

    std::uniform_real_distribution<float> dist2(-limit2, limit2);
    W2_ = Eigen::MatrixXf(d_ff_, d_model_);

    for (int i = 0; i < d_ff_; ++i){
        for (int j = 0; j < d_model_; ++j){
            W2_(i, j) = dist2(gen);
        }
    }

    b1_ = Eigen::VectorXf::Zero(d_ff_);
    b2_ = Eigen::VectorXf::Zero(d_model_);

}


Eigen::MatrixXf FeedForward::relu(const Eigen::MatrixXf& x){
    return x.cwiseMax(0.0f);
}

Eigen::MatrixXf FeedForward::forward(const Eigen::MatrixXf& x){
    last_input_ = x;

    Eigen::MatrixXf hidden = x * W1_;

    for (int i = 0; i < hidden.rows(); ++i){
        hidden.row(i) += b1_.transpose();
    }

    hidden = relu(hidden);

    last_hidden_ = hidden;

    Eigen::MatrixXf output = last_hidden_ * W2_;

    for (int i = 0; i < output.rows(); ++i){
        output.row(i) += b2_.transpose();
    }

    return output;
}


void FeedForward::update_parameters(const Eigen::MatrixXf& d_W1, const Eigen::VectorXf& d_b1,
const Eigen::MatrixXf& d_W2, const Eigen::VectorXf& d_b2){
    W1_ -= d_W1;
    b1_ -= d_b1;
    W2_ -= d_W2;
    b2_ -= d_b2;
}


}