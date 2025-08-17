#include <gtest/gtest.h>
#include "feed_forward.hpp"
#include <cmath>

class FeedForwardTest : public ::testing::Test {
    protected:
    void SetUp() override {
        d_model = 4;
        d_ff = 8;
        feed_forward = std::make_unique<transformer::FeedForward>(d_model, d_ff);
    }

    int d_model;
    int d_ff;
    std::unique_ptr<transformer::FeedForward> feed_forward;
};

TEST_F(FeedForwardTest, ConstructorTest) {
    EXPECT_EQ(feed_forward->get_d_model(), d_model);
    EXPECT_EQ(feed_forward->get_d_ff(), d_ff);

    EXPECT_EQ(feed_forward->get_W1().rows(), d_model);
    EXPECT_EQ(feed_forward->get_W1().cols(), d_ff);
    EXPECT_EQ(feed_forward->get_W2().rows(), d_ff);
    EXPECT_EQ(feed_forward->get_W2().cols(), d_model);
    EXPECT_EQ(feed_forward->get_b1().size(), d_ff);
    EXPECT_EQ(feed_forward->get_b2().size(), d_model);
}

TEST_F(FeedForwardTest, XavierInitializationTest) {
    const auto& W1 = feed_forward -> get_W1();
    const auto& W2 = feed_forward -> get_W2();

    float limit1 = std::sqrt(6.0f / (d_model + d_ff));
    for (int i = 0; i < d_model; ++i){
        for (int j = 0; j < d_ff; ++j){
            EXPECT_GE(W1(i, j), -limit1);
            EXPECT_LE(W1(i, j), limit1);
        }
    }

    float limit2 = std::sqrt(6.0f / (d_ff + d_model));

    for (int i = 0; i < d_ff; ++i){
        for (int j = 0; j < d_model; ++j){
            EXPECT_GE(W2(i, j), -limit2);
            EXPECT_LE(W2(i, j), limit2);
        }
    }

    for (int i = 0; i < feed_forward->get_b1().size(); ++i){
        EXPECT_FLOAT_EQ(feed_forward->get_b1()(i), 0.0f);
    }

    for (int i = 0; i < feed_forward->get_b2().size(); ++i){
        EXPECT_FLOAT_EQ(feed_forward->get_b2()(i), 0.0f);
    }
}

TEST_F(FeedForwardTest, ForwardPassShapeTest) {
    int seq_len = 3;
    Eigen::MatrixXf input(seq_len, d_model);

    input.setRandom();

    Eigen::MatrixXf output = feed_forward -> forward(input);

    EXPECT_EQ(output.rows(), seq_len);
    EXPECT_EQ(output.cols(), d_model);
}

TEST_F(FeedForwardTest, ReLUActivationTest) {
    transformer::FeedForward simple_ff(2, 2);

    Eigen::MatrixXf input(1, 2);

    input << 1.0f, -1.0f;

    Eigen::MatrixXf output = simple_ff.forward(input);

    const auto& hidden = simple_ff.get_last_hidden();
    for (int i = 0; i < hidden.rows(); ++i){
        for (int j = 0; j < hidden.cols(); ++j){
            EXPECT_GE(hidden(i, j), 0.0f);
        }
    }
}


TEST_F(FeedForwardTest, DifferentInputProduceDifferentOutputTest) {
    Eigen::MatrixXf input1(2, d_model);
    input1.setRandom();

    Eigen::MatrixXf input2(2, d_model);
    input2.setRandom();

    Eigen::MatrixXf output1 = feed_forward -> forward(input1);
    Eigen::MatrixXf output2 = feed_forward -> forward(input2);

    EXPECT_FALSE(output1.isApprox(output2));
}

TEST_F(FeedForwardTest, ParameterUpdateTest) {
    // Get initial parameters
    Eigen::MatrixXf initial_W1 = feed_forward->get_W1();
    Eigen::VectorXf initial_b1 = feed_forward->get_b1();
    Eigen::MatrixXf initial_W2 = feed_forward->get_W2();
    Eigen::VectorXf initial_b2 = feed_forward->get_b2();
    
    // Create some dummy gradients
    Eigen::MatrixXf d_W1 = Eigen::MatrixXf::Constant(d_model, d_ff, 0.1f);
    Eigen::VectorXf d_b1 = Eigen::VectorXf::Constant(d_ff, 0.1f);
    Eigen::MatrixXf d_W2 = Eigen::MatrixXf::Constant(d_ff, d_model, 0.1f);
    Eigen::VectorXf d_b2 = Eigen::VectorXf::Constant(d_model, 0.1f);
    
    // Update parameters
    feed_forward->update_parameters(d_W1, d_b1, d_W2, d_b2);
    
    // Check that parameters have changed
    EXPECT_FALSE(feed_forward->get_W1().isApprox(initial_W1, 1e-6f));
    EXPECT_FALSE(feed_forward->get_b1().isApprox(initial_b1, 1e-6f));
    EXPECT_FALSE(feed_forward->get_W2().isApprox(initial_W2, 1e-6f));
    EXPECT_FALSE(feed_forward->get_b2().isApprox(initial_b2, 1e-6f));
    
    // Check that the update was applied correctly (subtraction)
    EXPECT_TRUE(feed_forward->get_W1().isApprox(initial_W1 - d_W1, 1e-6f));
    EXPECT_TRUE(feed_forward->get_b1().isApprox(initial_b1 - d_b1, 1e-6f));
    EXPECT_TRUE(feed_forward->get_W2().isApprox(initial_W2 - d_W2, 1e-6f));
    EXPECT_TRUE(feed_forward->get_b2().isApprox(initial_b2 - d_b2, 1e-6f));
}

TEST_F(FeedForwardTest, ZeroInputTest) {
    // Test with zero input
    Eigen::MatrixXf zero_input = Eigen::MatrixXf::Zero(2, d_model);
    Eigen::MatrixXf output = feed_forward->forward(zero_input);
    
    // Output should not be all zeros due to bias terms (unless biases are also zero)
    // But at least the computation should complete without errors
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), d_model);
}

