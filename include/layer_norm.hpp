#pragma once
#include <Eigen/Dense>

namespace transformer {

/**
 * @brief Layer Normalization
 * Normalizes inputs across the feature dimension with learnable parameters
 */
class LayerNorm {
private:
    int d_model_;
    float epsilon_;

    Eigen::VectorXf gamma_; // Scale parameter (initialized to 1)
    Eigen::VectorXf beta_;  // Shift parameter (initialized to 0)

    // For storing intermediate values (useful for backpropagation later)
    Eigen::MatrixXf last_input_;
    Eigen::MatrixXf last_normalized_;
    Eigen::VectorXf last_mean_;      // ✅ Fixed: VectorXf with underscore
    Eigen::VectorXf last_variance_;  // ✅ Fixed: VectorXf with underscore

public:
    LayerNorm(int d_model, float epsilon = 1e-6f);

    /**
     * @brief Forward pass through layer normalization
     * @param x Input matrix of shape (seq_len, d_model)
     * @return Normalized output of shape (seq_len, d_model)
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);

    /**
     * @brief Initialize parameters (gamma = 1, beta = 0)
     */
    void initialize_parameters();

    /**
     * @brief Get learnable parameters
     */
    const Eigen::VectorXf& get_gamma() const { return gamma_; }  // ✅ Fixed: const reference
    const Eigen::VectorXf& get_beta() const { return beta_; }   // ✅ Fixed: const reference

    /** 
     * @brief Update parameters  // ✅ Fixed: typo
     */
    void update_parameters(const Eigen::VectorXf& dgamma, const Eigen::VectorXf& dbeta);

    /**
     * @brief Get last input for backpropagation
     */
    const Eigen::MatrixXf& get_last_input() const { return last_input_; }
    const Eigen::MatrixXf& get_last_normalized() const { return last_normalized_; }
    const Eigen::VectorXf& get_last_mean() const { return last_mean_; }          // ✅ Fixed: VectorXf
    const Eigen::VectorXf& get_last_variance() const { return last_variance_; }  // ✅ Fixed: VectorXf

    /**
     * @brief Set epsilon
     */
    void set_epsilon(float epsilon) { epsilon_ = epsilon; }

    /**
     * @brief Get epsilon
     */
    float get_epsilon() const { return epsilon_; }

    /**
     * @brief Get d_model
     */
    int get_d_model() const { return d_model_; }
};  // ✅ Fixed: Added semicolon

} // namespace transformer