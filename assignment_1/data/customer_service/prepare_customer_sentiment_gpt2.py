import os
import pandas as pd
import tiktoken
import numpy as np

# ----------------------------------
# Paths
input_csv_path = 'train.csv'
output_dir = 'gpt2_prepared'
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------
# Load and join all text
df = pd.read_csv(input_csv_path)
if 'conversation' not in df.columns:
    raise ValueError("CSV must contain a 'conversation' column.")

texts = df['conversation'].dropna().astype(str).tolist()
full_text = "\n".join(texts)

# Optional: Save raw text for inspection
with open(os.path.join(output_dir, 'dataset.txt'), 'w', encoding='utf-8') as f:
    f.write(full_text)

# ----------------------------------
# Train/val split
n = len(full_text)
train_text = full_text[:int(n * 0.9)]
val_text = full_text[int(n * 0.9):]

# ----------------------------------
# Tokenize using GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_text)
val_ids = enc.encode_ordinary(val_text)

print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens:   {len(val_ids):,}")

# ----------------------------------
# Save as .bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))
