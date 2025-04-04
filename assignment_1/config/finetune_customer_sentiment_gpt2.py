import time

# Output folder for logs & checkpoints
out_dir = 'out-customer-gpt2'

# Evaluation & logging
eval_interval = 5
eval_iters = 40

# WandB
wandb_log = True
wandb_project = 'customer-sentiment'
wandb_run_name = 'gpt2-ft-' + str(time.time())

# Dataset
dataset = 'gpt2_prepared'  # name of the folder inside data/
init_from = 'gpt2'          # use pre-trained GPT-2 weights

# Checkpoint saving strategy
always_save_checkpoint = True  # only save if val loss improves

# Training strategy
batch_size = 1
gradient_accumulation_steps = 32  # simulates 32 effective batch size
max_iters = 40               # you can increase if needed

# Learning rate
learning_rate = 1e-5
decay_lr = False  # keep LR constant for fine-tuning

# System
device = 'mps'  # or 'mps' or 'cpu'
compile = False
