# Find Google Test
find_package(GTest REQUIRED)

# Create test executable
add_executable(embedding_tests
    test_embedding.cpp
)

# Link libraries
target_link_libraries(embedding_tests 
    transformer_lib 
    GTest::gtest 
    GTest::gtest_main
)

# Enable testing
enable_testing()
add_test(NAME EmbeddingTests COMMAND embedding_tests)