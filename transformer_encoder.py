import numpy as np
import pandas as pd

np.random.seed(42)

vocab = {"o": 0, "banco": 1, "bloqueou": 2, "cartao": 3}
vocab_df = pd.DataFrame(list(vocab.items()), columns=["palavra", "id"])

frase = ["o", "banco", "bloqueou", "o", "cartao"]
ids = [vocab[p] for p in frase]

d_model = 64
d_ff = d_model * 4
vocab_size = len(vocab)
N = 6

embedding_table = np.random.randn(vocab_size, d_model)

X = embedding_table[ids]
X = X[np.newaxis, :, :]


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V


def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def feed_forward(x, W1, b1, W2, b2):
    hidden = np.maximum(0, x @ W1 + b1)
    return hidden @ W2 + b2


def init_encoder_layer():
    return {
        "W_q": np.random.randn(d_model, d_model),
        "W_k": np.random.randn(d_model, d_model),
        "W_v": np.random.randn(d_model, d_model),
        "W1": np.random.randn(d_model, d_ff),
        "b1": np.zeros(d_ff),
        "W2": np.random.randn(d_ff, d_model),
        "b2": np.zeros(d_model),
    }


def encoder_layer(X, params):
    X_att = scaled_dot_product_attention(X, params["W_q"], params["W_k"], params["W_v"])
    X_norm1 = layer_norm(X + X_att)
    X_ffn = feed_forward(X_norm1, params["W1"], params["b1"], params["W2"], params["b2"])
    X_out = layer_norm(X_norm1 + X_ffn)
    return X_out


layers = [init_encoder_layer() for _ in range(N)]

Z = X
for i, params in enumerate(layers):
    Z = encoder_layer(Z, params)

print("Vocabulary:")
print(vocab_df)
print(f"\nSentence: {frase}")
print(f"IDs: {ids}")
print(f"\nInput tensor X shape: {X.shape}")
print(f"Output vector Z shape after {N} encoder layers: {Z.shape}")
print("\nShape preserved:", X.shape == Z.shape)
