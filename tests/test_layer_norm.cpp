#include <gtest/gtest.h>
#include <layer_norm.hpp>
#include <vector>
#include <cmath>
#include <memory>

class LayerNormTest : public ::testing::Test {
    protected:
    void SetUp() override {
        d_model = 4;
        layer_norm = std::make_unique<transformer::LayerNorm>(d_model, 1e-6f);
    }

    int d_model;
    std::unique_ptr<transformer::LayerNorm> layer_norm;
};

TEST_F(LayerNormTest, ConstructorTest){

    auto gamma = layer_norm->get_gamma();
    auto beta = layer_norm->get_beta();

    for (int i = 0; i < d_model; ++i){
        EXPECT_FLOAT_EQ(gamma(i), 1.0f);
        EXPECT_FLOAT_EQ(beta(i), 0.0f);
    }
}

TEST_F(LayerNormTest, NormalizationTest){
    Eigen::MatrixXf input(2, 4);
    input << 1, 2, 3, 4, 5, 6, 7, 8;

    auto output = layer_norm -> forward(input);

    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 4);

    for (int i = 0; i < 2; ++i){
        float row_mean = output.row(i).mean();
        float row_variance = (output.row(i).array() - row_mean).square().mean();

        EXPECT_NEAR(row_mean, 0.0f, 1e-5f);
        EXPECT_NEAR(row_variance, 1.0f, 1e-5f);
    }
}

TEST_F(LayerNormTest, EpsilonTest){
    Eigen::MatrixXf input = Eigen::MatrixXf::Zero(1, 4);

    auto output = layer_norm -> forward(input);

    for (int i = 0; i < output.rows(); ++i){
        for (int j = 0; j < output.cols(); ++j){
            EXPECT_TRUE(std::isfinite(output(i, j)));
        }
    }
}

int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}



