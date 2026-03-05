# Transformer Encoder From Scratch

Implementation of the forward pass of a complete Transformer Encoder, based on the paper "Attention Is All You Need" (Vaswani et al., 2017).

## Requirements

- Python 3.x
- numpy
- pandas

## How to run

```bash
python3 transformer_encoder.py
```

## What it does

1. Creates a simple vocabulary and maps an input sentence to integer IDs.
2. Initializes a random embedding table and builds the input tensor X with shape `(BatchSize, SequenceLength, d_model)`.
3. Implements the Encoder sub-layers:
   - **Scaled Dot-Product Attention**: computes Q, K, V from linear projections and applies scaled softmax.
   - **Add & LayerNorm**: residual connection followed by layer normalization.
   - **Feed-Forward Network (FFN)**: two linear transformations with ReLU activation.
4. Stacks N=6 identical Encoder layers, producing the contextualized representation vector Z.

## Expected output

The input tensor `X` with shape `(1, 5, 64)` passes through all 6 layers and comes out as `Z` with the same dimensions, but with contextualized token representations.
