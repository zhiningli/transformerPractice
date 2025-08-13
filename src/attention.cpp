#include "attention.hpp"
#include <iostream>

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

}
