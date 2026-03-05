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

print("Vocabulary:")
print(vocab_df)
print(f"\nSentence: {frase}")
print(f"IDs: {ids}")
print(f"\nInput tensor X shape: {X.shape}")
print("(BatchSize, SequenceLength, d_model)")
