"""
Prepare the Customer Service Sentiment dataset for character-level language modeling.
Instead of using BPE tokens, we map characters to integer IDs.
Saves: train.bin, val.bin, and meta.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd

# Output directory
output_dir = os.path.join(os.path.dirname(__file__), 'processed_data/customer_sentiment')
os.makedirs(output_dir, exist_ok=True)

# Load CSV
df = pd.read_csv('./train.csv')

# Keep only the conversation and sentiment columns
df = df[['conversation', 'customer_sentiment']]
df.dropna(inplace=True)

# Combine into single text: conversation + sentiment marker
data = df['conversation'] + " <|" + df['customer_sentiment'] + "|>"
full_text = "\n".join(data.tolist())

# Build vocabulary
chars = sorted(list(set(full_text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")
print("Characters:", ''.join(chars))

# Char-level encoding
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Train/val split (90/10)
n = len(full_text)
train_text = full_text[:int(n * 0.9)]
val_text = full_text[int(n * 0.9):]

train_ids = np.array(encode(train_text), dtype=np.uint16)
val_ids = np.array(encode(val_text), dtype=np.uint16)

# Save binary files
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

# Save meta info
meta = {
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos,
}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"Saved dataset to: {output_dir}")
print(f"Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids):,}")
