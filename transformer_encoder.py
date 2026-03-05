import numpy as np
import pandas as pd

np.random.seed(42)

vocab = {"o": 0, "banco": 1, "bloqueou": 2, "cartao": 3}
vocab_df = pd.DataFrame(list(vocab.items()), columns=["palavra", "id"])

frase = ["o", "banco", "bloqueou", "o", "cartao"]
ids = [vocab[p] for p in frase]

d_model = 64
vocab_size = len(vocab)

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


W_q = np.random.randn(d_model, d_model)
W_k = np.random.randn(d_model, d_model)
W_v = np.random.randn(d_model, d_model)

X_att = scaled_dot_product_attention(X, W_q, W_k, W_v)
X_norm1 = layer_norm(X + X_att)

print("Vocabulary:")
print(vocab_df)
print(f"\nSentence: {frase}")
print(f"IDs: {ids}")
print(f"\nInput tensor X shape: {X.shape}")
print(f"After Add & LayerNorm: {X_norm1.shape}")
