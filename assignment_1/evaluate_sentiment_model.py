import torch
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model import GPTConfig, GPT
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------
# Paths
ckpt_path = 'out-customer-sentiment/ckpt.pt'
meta_path = 'data/customer_service/processed_data/customer_sentiment/meta.pkl'
test_csv_path = 'data/customer_service/test.csv'

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
label_map = {0: "negative", 1: "neutral", 2: "positive"}
inv_label_map = {v: k for k, v in label_map.items()}

# ----------------------------------
# Load vocab
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi = meta['stoi']

def encode(text):
    return [stoi.get(c, 0) for c in text][:64]

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
def predict(text):
    input_ids = torch.tensor(encode(str(text)), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(input_ids)
        pred = torch.argmax(logits, dim=-1).item()
    return pred

# ----------------------------------
# Load test data
df = pd.read_csv(test_csv_path)

if 'conversation' not in df.columns or 'customer_sentiment' not in df.columns:
    raise ValueError("CSV must contain 'conversation' and 'customer_sentiment' columns.")

# Predict
y_true = df['customer_sentiment'].map(inv_label_map).tolist()
y_pred = df['conversation'].apply(predict).tolist()

# Metrics
acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {acc:.4f}\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"]))


# print("Confusion Matrix:")
# print(confusion_matrix(y_true, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
labels = ["negative", "neutral", "positive"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

