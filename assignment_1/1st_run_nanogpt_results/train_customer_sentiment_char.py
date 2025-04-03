# Fine-tuning GPT for Customer Sentiment Classification (Char-level)

out_dir = 'out-customer-sentiment'
eval_interval = 500
eval_iters = 20
log_interval = 10

always_save_checkpoint = True  # save progress

wandb_log = True
wandb_project = 'customer-sentiment'
wandb_run_name = 'nanoGPT'

# Custom dataset name (used by train.py to load meta/vocab)
dataset = 'customer_sentiment'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128  # length of tokenized conversation

# Small GPT config (you can increase this if you want better results)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

# Optimizer config
learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# System
device = 'mps'  # or 'mps' or 'cpu'
compile = False
