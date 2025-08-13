#include <iostream>
#include <vector>
#include "embedding.hpp"

int main() {
    std::cout << "Transformer Practice - Learning C++ and Transformer Architecture!" << std::endl;
    std::cout << "This project will be built step by step." << std::endl;
    
    std::cout << "Testing Token Embedding" << std::endl;
    
    transformer::TokenEmbedding token_embedding(10, 4);

    std::vector<int> token_indices = {0, 2, 5, 1};

    Eigen::MatrixXf embeddings = token_embedding.forward(token_indices);

    std::cout << "Input token indices: ";
    for (int idx : token_indices){
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    std::cout << "Output Embedding matrix:" << std::endl;
    std::cout << embeddings << std::endl;

    // Test that embeddings are different for different tokens
    std::cout << "\nTesting that different tokens have different embeddings:" << std::endl;
    Eigen::MatrixXf emb1 = token_embedding.forward({0});
    Eigen::MatrixXf emb2 = token_embedding.forward({1});
    std::cout << "Token 0 embedding: " << emb1.row(0) << std::endl;
    std::cout << "Token 1 embedding: " << emb2.row(0) << std::endl;
    std::cout << "Are they different? " << (emb1 != emb2 ? "Yes!" : "No!") << std::endl;
    
    return 0;
} 