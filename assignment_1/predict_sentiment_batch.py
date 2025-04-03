import torch
import pandas as pd
import pickle
from model import GPTConfig, GPT

# ----------------------------------
# Paths
ckpt_path = 'out-customer-sentiment/ckpt.pt'
meta_path = 'data/customer_service/processed_data/customer_sentiment/meta.pkl'
test_csv_path = 'data/customer_service/test.csv'  # <== points to the uploaded file
output_csv_path = 'test_with_predictions.csv'

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
label_map = {0: "negative", 1: "neutral", 2: "positive"}

# ----------------------------------
# Load vocab and tokenizer
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi = meta['stoi']

def encode(text):
    return [stoi.get(c, 0) for c in text][:128]  # truncate to model's block_size

# ----------------------------------
# Load model
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# ----------------------------------
# Predict function
def predict_sentiment(text):
    input_ids = torch.tensor(encode(str(text)), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(input_ids)
        prediction = torch.argmax(logits, dim=-1).item()
    return label_map[prediction]

# ----------------------------------
# Run batch prediction
df = pd.read_csv(test_csv_path)

if 'conversation' not in df.columns:
    raise ValueError("CSV must contain a 'conversation' column.")

df['predicted_sentiment'] = df['conversation'].apply(predict_sentiment)
df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to: {output_csv_path}")
