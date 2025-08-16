# Find Google Test
find_package(GTest REQUIRED)

# Create test executable
add_executable(embedding_tests test_embedding.cpp)
add_executable(attention_tests test_attention.cpp)
add_executable(multihead_attention_tests test_multihead_attention.cpp)

# Link libraries
target_link_libraries(embedding_tests transformer_lib GTest::gtest GTest::gtest_main)
target_link_libraries(attention_tests transformer_lib GTest::gtest GTest::gtest_main)
target_link_libraries(multihead_attention_tests transformer_lib GTest::gtest GTest::gtest_main)

# Enable testing
enable_testing()
add_test(NAME EmbeddingTests COMMAND embedding_tests)
add_test(NAME AttentionTests COMMAND attention_tests)
add_test(NAME MultiHeadAttentionTests COMMAND multihead_attention_tests)