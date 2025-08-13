#include <gtest/gtest.h>
#include "attention.hpp"
#include <vector>
#include <cmath>
#include <memory>

class ScaledDotProductAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        d_k = 4;
        d_v = 4;
        seq_len = 3;
        attention = std::make_unique<transformer::ScaledDotProductAttention>(d_k);
    }

    int d_k;
    int d_v;
    int seq_len;
    std::unique_ptr<transformer::ScaledDotProductAttention> attention;
};

TEST_F(ScaledDotProductAttentionTest, ConstructorTest) {
    // Test that scale factor is correctly computed
    float expected_scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    EXPECT_FLOAT_EQ(attention->get_scale_factor(), expected_scale);
}

TEST_F(ScaledDotProductAttentionTest, BasicForwardTest) {
    // Create simple Q, K, V matrices
    Eigen::MatrixXf Q(seq_len, d_k);
    Eigen::MatrixXf K(seq_len, d_k);
    Eigen::MatrixXf V(seq_len, d_v);
    
    // Initialize with simple values
    Q << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0;
         
    K << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0;
         
    V << 0.5, 0.2, 0.1, 0.3,
         0.3, 0.8, 0.4, 0.1,
         0.7, 0.4, 0.9, 0.2;

    auto result = attention->forward(Q, K, V);
    
    // Check output shape
    EXPECT_EQ(result.rows(), seq_len);
    EXPECT_EQ(result.cols(), d_v);
}

TEST_F(ScaledDotProductAttentionTest, SelfAttentionTest) {
    // Test with a scenario where one query strongly matches one key
    Eigen::MatrixXf Q(3, d_k);
    Eigen::MatrixXf K(3, d_k);
    Eigen::MatrixXf V(3, d_v);
    
    // First query matches first key very strongly
    Q << 10, 0, 0, 0,   // Strong signal
         0, 0.1, 0, 0,  // Weak signal
         0, 0, 0.1, 0;  // Weak signal
         
    K << 10, 0, 0, 0,   // Matches first query
         0, 1, 0, 0,    // Different pattern
         0, 0, 1, 0;    // Different pattern
         
    V << 100, 200, 300, 400,  // First value
         10, 20, 30, 40,      // Second value  
         1, 2, 3, 4;          // Third value

    auto result = attention->forward(Q, K, V);
    
    // First token should attend primarily to itself (first value)
    // So result[0] should be close to V[0]
    for (int j = 0; j < d_v; ++j) {
        EXPECT_NEAR(result(0, j), V(0, j), 10.0f); // Allow some tolerance
    }
}

TEST_F(ScaledDotProductAttentionTest, UniformAttentionTest) {
    // When all queries are identical and all keys are identical,
    // attention should be uniform across all values
    Eigen::MatrixXf Q(3, d_k);
    Eigen::MatrixXf K(3, d_k);
    Eigen::MatrixXf V(3, d_v);
    
    // All queries and keys are the same
    Q.setOnes();
    K.setOnes();
    
    V << 1, 2, 3, 4,
         5, 6, 7, 8,
         9, 10, 11, 12;

    auto result = attention->forward(Q, K, V);
    
    // All output rows should be the same (average of all V rows)
    Eigen::VectorXf expected_output = V.colwise().mean();
    
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_v; ++j) {
            EXPECT_NEAR(result(i, j), expected_output(j), 1e-5);
        }
    }
}

TEST_F(ScaledDotProductAttentionTest, ScaleFactorTest) {
    // Test different d_k values
    transformer::ScaledDotProductAttention attention_small(4);
    transformer::ScaledDotProductAttention attention_large(16);
    
    float expected_small = 1.0f / std::sqrt(4.0f);   // = 0.5
    float expected_large = 1.0f / std::sqrt(16.0f);  // = 0.25
    
    EXPECT_FLOAT_EQ(attention_small.get_scale_factor(), expected_small);
    EXPECT_FLOAT_EQ(attention_large.get_scale_factor(), expected_large);
    
    // FIXED: Smaller d_k should have LARGER scale factor
    EXPECT_GT(attention_small.get_scale_factor(), attention_large.get_scale_factor());
}

TEST_F(ScaledDotProductAttentionTest, RandomInputTest) {
    // Test with random inputs to ensure no crashes
    Eigen::MatrixXf Q(seq_len, d_k);
    Eigen::MatrixXf K(seq_len, d_k);
    Eigen::MatrixXf V(seq_len, d_v);
    
    Q.setRandom();
    K.setRandom();
    V.setRandom();
    
    auto result = attention->forward(Q, K, V);
    
    // Should produce valid output
    EXPECT_EQ(result.rows(), seq_len);
    EXPECT_EQ(result.cols(), d_v);
    
    // All values should be finite (not NaN or inf)
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            EXPECT_TRUE(std::isfinite(result(i, j)));
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}