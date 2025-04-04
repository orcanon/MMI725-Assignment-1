import torch
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from model import GPTConfig, GPT
import tiktoken
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend
import seaborn as sns

# ----------------------------------
# Paths
ckpt_path = 'out-customer-gpt2/ckpt.pt'
test_csv_path = 'data/customer_service/test.csv'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

label_map = {0: "negative", 1: "neutral", 2: "positive"}
inv_label_map = {v: k for k, v in label_map.items()}

# ----------------------------------
# Tokenizer (GPT-2 BPE)
enc = tiktoken.get_encoding("gpt2")

def encode(text, max_length=2048):
    return enc.encode_ordinary(str(text))[:max_length]

# ----------------------------------
# Dataset for classification
class CustomerSentimentDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.texts = df['conversation'].astype(str).tolist()
        self.labels = df['customer_sentiment'].map(inv_label_map).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = encode(self.texts[idx])
        x = torch.tensor(input_ids, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return inputs, labels

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
def predict_batch(model, loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            logits, _ = model(X, Y)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(Y.cpu().tolist())
    return all_labels, all_preds

# ----------------------------------
# Run evaluation
test_dataset = CustomerSentimentDataset(test_csv_path)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

y_true, y_pred = predict_batch(model, test_loader)

acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {acc:.4f}\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(label_map.values()), zero_division=0))

cm = confusion_matrix(y_true, y_pred)
labels = list(label_map.values())

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("✅ Confusion matrix saved to: confusion_matrix.png")
