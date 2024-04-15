# Transformer_Scratch
This is the code of Transformer Architecture from Scratch
Here's a README file for the provided Python code which implements a Transformer model. This README outlines the components of the Transformer architecture, the purpose of each class, and some basic usage instructions.

---

# Transformer Model Implementation in PyTorch

This Python module implements the Transformer architecture as described in the paper "Attention is All You Need" by Vaswani et al. The implementation is done using PyTorch and includes both the encoder and decoder components.

## Components

The Transformer model includes several key components, each defined as a Python class:

- **`MultiHeadAttention`**: This class implements the multi-head attention mechanism which allows the model to jointly attend to information from different representation subspaces at different positions.

- **`PositionwiseFeedForward`**: A simple feed-forward network applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

- **`positionEncoding`**: This module injects some information about the relative or absolute position of the tokens in the sequence. The positional encodings have the same dimension as the embeddings, so that the two can be summed.

- **`EncoderLayer`**: Each encoder layer consists of a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. This class also applies layer normalization and dropout.

- **`DecoderLayer`**: Similar to the encoder layer, but in addition to the two sub-layers, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.

- **`Transformer`**: The full Transformer model which stacks multiple encoder and decoder layers and provides the interface for processing input and generating output.

## Usage

The `Transformer` class is the main model which can be instantiated and used directly in a training or inference pipeline. Here's a simple example of how to instantiate and use the Transformer:

```python
from model import Transformer

# Configuration
src_vocab_size = 8000
tgt_vocab_size = 8000
num_layers = 6
num_heads = 8
d_model = 512
d_ff = 2048
max_seq_len = 100
dropout_rate = 0.1

# Instantiate the model
model = Transformer(
    src_vocab_size,
    tgt_vocab_size,
    num_layers,
    num_heads,
    d_model,
    d_ff,
    max_seq_len,
    dropout_rate
)

# Example input (batch size, sequence length)
src = torch.randint(0, src_vocab_size, (10, 50))
tgt = torch.randint(0, tgt_vocab_size, (10, 50))

# Forward pass
output = model(src, tgt)
```

## Installation

To use this module, you need to have Python and PyTorch installed. The code is compatible with PyTorch 1.x versions.

1. Install PyTorch according to the instructions on the official website: https://pytorch.org/get-started/locally/
2. Clone the repository containing this module.
3. Import and use the module as demonstrated above.

