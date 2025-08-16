
#include <gtest/gtest.h>
#include "attention.hpp"
#include <vector>
#include <cmath>
#include <memory>

class MultiHeadAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        num_heads = 2;
        d_model = 8;
        seq_len = 3;
        attention = std::make_unique<transformer::MultiHeadAttention>(num_heads, d_model);
    }

    int num_heads;
    int d_model;
    int seq_len;
    std::unique_ptr<transformer::MultiHeadAttention> attention;
};

TEST_F(MultiHeadAttentionTest, ConstructorTest) {
    // Test that constructor works and d_k calculation is correct
    float expected_scale = 1.0f / std::sqrt(static_cast<float>(d_model / num_heads));
    EXPECT_FLOAT_EQ(attention->get_scale_factor(), expected_scale);
}

TEST_F(MultiHeadAttentionTest, InvalidDimensionTest) {
    // Test that constructor throws when d_model not divisible by num_heads
    EXPECT_THROW(transformer::MultiHeadAttention(3, 8), std::invalid_argument);
}

TEST_F(MultiHeadAttentionTest, BasicForwardTest) {
    // Test basic forward pass
    Eigen::MatrixXf query(seq_len, d_model);
    Eigen::MatrixXf key(seq_len, d_model);
    Eigen::MatrixXf value(seq_len, d_model);
    
    // Initialize with simple values
    query.setOnes();
    key.setOnes();
    value.setOnes();
    
    auto result = attention->forward(query, key, value);
    
    // Check output shape
    EXPECT_EQ(result.rows(), seq_len);
    EXPECT_EQ(result.cols(), d_model);
    
    // Check that output is finite (no NaN or inf)
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            EXPECT_TRUE(std::isfinite(result(i, j)));
        }
    }
}

TEST_F(MultiHeadAttentionTest, SelfAttentionTest) {
    // Test self-attention (query = key = value)
    Eigen::MatrixXf input(seq_len, d_model);
    input.setRandom();
    
    auto result = attention->forward(input, input, input);
    
    // Output should have same shape as input
    EXPECT_EQ(result.rows(), input.rows());
    EXPECT_EQ(result.cols(), input.cols());
}

TEST_F(MultiHeadAttentionTest, DifferentInputsTest) {
    // Test that different inputs produce different outputs
    Eigen::MatrixXf input1(seq_len, d_model);
    Eigen::MatrixXf input2(seq_len, d_model);
    
    input1.setOnes();
    input2.setZero();
    
    auto result1 = attention->forward(input1, input1, input1);
    auto result2 = attention->forward(input2, input2, input2);
    
    // Results should be different
    EXPECT_FALSE(result1.isApprox(result2, 1e-6));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}