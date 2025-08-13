# Transformer Practice - Learning C++ and Transformer Architecture

This project is designed to help myself learn both C++ and transformer architecture by building a complete transformer from scratch, step by step.

## Project Structure

```
transformerPractice/
├── CMakeLists.txt          # Main build configuration
├── include/                # Header files (.hpp)
│   ├── embedding.hpp       # Token and positional embeddings
│   ├── attention.hpp       # Attention mechanisms
│   ├── feed_forward.hpp    # Feed-forward networks
│   ├── layer_norm.hpp      # Layer normalization
│   ├── transformer_block.hpp # Encoder/Decoder blocks
│   └── transformer.hpp     # Complete transformer
├── src/                    # Implementation files (.cpp)
│   ├── CMakeLists.txt      # Library build configuration
│   ├── embedding.cpp
│   ├── attention.cpp
│   ├── feed_forward.cpp
│   ├── layer_norm.cpp
│   ├── transformer_block.cpp
│   └── transformer.cpp
├── examples/               # Example programs
│   ├── CMakeLists.txt
│   └── main.cpp           # Main demo program
├── tests/                  # Unit tests (future)
│   └── CMakeLists.txt
└── README.md              # This file
```

## Learning Path

We'll build the transformer in this order:

1. **Embedding Layer** - Token embeddings and positional encoding
2. **Attention Mechanism** - Scaled dot-product attention
3. **Multi-Head Attention** - Multiple attention heads
4. **Feed-Forward Network** - Position-wise feed-forward layers
5. **Layer Normalization** - Normalization layers
6. **Transformer Blocks** - Encoder and decoder blocks
7. **Complete Transformer** - Full encoder-decoder architecture

## Prerequisites

- C++17 compatible compiler
- CMake 3.16 or higher
- Eigen3 library

## Building the Project

```bash
mkdir build
cd build
cmake ..
make
```

## Where to Start

Start with **Step 1: Embedding Layer**. We'll create the header file first, then implement it, and test it before moving to the next component.

Ready to begin? Let's start with the embedding layer! 