# Create transformer library
add_library(transformer_lib STATIC
    embedding.cpp
    attention.cpp
    layer_norm.cpp
)

# Link Eigen3
target_link_libraries(transformer_lib Eigen3::Eigen) 