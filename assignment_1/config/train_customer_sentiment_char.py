# Fine-tuning GPT for Customer Sentiment Classification (Char-level)

out_dir = 'out-customer-sentiment'
eval_interval = 250
eval_iters = 20
log_interval = 10

always_save_checkpoint = True  # save progress

wandb_log = True
wandb_project = 'customer-sentiment'
wandb_run_name = 'nanoGPT'

# Custom dataset name (used by train.py to load meta/vocab)
dataset = 'processed_data/customer_sentiment'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 64  # length of tokenized conversation

# Small GPT config (you can increase this if you want better results)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# Optimizer config
learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# System
device = 'mps'  # or 'mps' or 'cpu'
compile = False
