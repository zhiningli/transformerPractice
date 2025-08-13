
#include <gtest/gtest.h>
#include "embedding.hpp"
#include <vector>
#include <cmath>
#include <memory> 

class TokenEmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override {
        vocab_size = 100;
        embedding_dim = 8;
        embedding = std::make_unique<transformer::TokenEmbedding>(vocab_size, embedding_dim);
    }

    int vocab_size;
    int embedding_dim;
    std::unique_ptr<transformer::TokenEmbedding> embedding;
};

TEST_F(TokenEmbeddingTest, ConstructorTest) {
    auto matrix = embedding->get_embedding_matrix();
    EXPECT_EQ(matrix.rows(), vocab_size);
    EXPECT_EQ(matrix.cols(), embedding_dim);
}

TEST_F(TokenEmbeddingTest, ForwardTest) {
    std::vector<int> token_indices = {1, 2, 3, 4, 5};
    auto result = embedding->forward(token_indices);

    EXPECT_EQ(result.rows(), token_indices.size());
    EXPECT_EQ(result.cols(), embedding_dim);

    auto matrix = embedding->get_embedding_matrix();
    for (int i = 0; i < token_indices.size(); i++) {
        EXPECT_EQ(result.row(i), matrix.row(token_indices[i]));
    }
}

TEST_F(TokenEmbeddingTest, DifferentTokensHaveDifferentEmbeddings) {
    auto emb1 = embedding->forward({0});
    auto emb2 = embedding->forward({1});

    EXPECT_NE(emb1.row(0), emb2.row(0));
}

TEST_F(TokenEmbeddingTest, OutOfRangeTest) {
    std::vector<int> invalid_indices = {100, 101, 102};
    EXPECT_THROW(embedding->forward(invalid_indices), std::out_of_range);
}

class PositionalEncodingTest : public ::testing::Test {
protected:
    void SetUp() override {
        max_seq_len = 100;
        embedding_dim = 8;
        pos_encoding = std::make_unique<transformer::PositionalEncoding>(max_seq_len, embedding_dim);
    }

    int max_seq_len;
    int embedding_dim;
    std::unique_ptr<transformer::PositionalEncoding> pos_encoding;
};

TEST_F(PositionalEncodingTest, ConstructorTest) {
    auto encoding_matrix = pos_encoding->get_pos_encoding();
    EXPECT_EQ(encoding_matrix.rows(), max_seq_len);
    EXPECT_EQ(encoding_matrix.cols(), embedding_dim);
}

TEST_F(PositionalEncodingTest, SinusoidalPatternTest) {
    auto encoding_matrix = pos_encoding->get_pos_encoding();

    for (int pos = 0; pos < std::min(10, max_seq_len); pos++) {
        for (int i = 0; i < embedding_dim; i++) {
            float expected_value;
            float denominator = std::pow(10000.0f, (2.0f * i) / embedding_dim);
            float angle = pos / denominator;

            if (i % 2 == 0) {
                expected_value = std::sin(angle);
            } else {
                expected_value = std::cos(angle);
            }
            
            EXPECT_NEAR(encoding_matrix(pos, i), expected_value, 1e-6);
        }
    }
}

TEST_F(PositionalEncodingTest, ForwardTest) {
    Eigen::MatrixXf token_embeddings(3, embedding_dim);
    token_embeddings.setRandom();

    auto result = pos_encoding->forward(token_embeddings);

    EXPECT_EQ(result.rows(), 3);
    EXPECT_EQ(result.cols(), embedding_dim);

    auto encoding_matrix = pos_encoding->get_pos_encoding();
    Eigen::MatrixXf expected = token_embeddings + encoding_matrix.block(0, 0, 3, embedding_dim);

    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.cols(); ++j) {
            EXPECT_NEAR(result(i, j), expected(i, j), 1e-6);
        }
    }
}

TEST_F(PositionalEncodingTest, OutOfRangeTest) {
    Eigen::MatrixXf large_embeddings(max_seq_len + 1, embedding_dim);
    EXPECT_THROW(pos_encoding->forward(large_embeddings), std::out_of_range);
}

TEST_F(PositionalEncodingTest, DifferentSequencesHaveDifferentEmbeddings) {
    auto encoding_matrix = pos_encoding->get_pos_encoding();

    EXPECT_NE(encoding_matrix.row(0), encoding_matrix.row(1));
    EXPECT_NE(encoding_matrix.row(1), encoding_matrix.row(2));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}