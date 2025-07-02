import math
from pathlib import Path
import time
import typing

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import torch
from torch.nn import functional as F

from dataloader import DataLoader
from gpt2 import GPT
from lr_scheduler import GPT2CosineLearningRateScheduler


if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Using device: {device}')

torch.set_float32_matmul_precision('high')

batch_size = 8
block_size = 1024
vocab_size=50_304
n_layer = 12
n_head = 12
n_embd = 768

data_loader = DataLoader(
    batch_size=batch_size,
    block_size=block_size,
    filepath=Path(__file__).parent / 'tiny-shakespeare.txt',
    device=device
)

model: GPT = GPT(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_head,
    block_size=block_size,
    vocab_size=vocab_size,
    flash_attention=True
)
model.to(device)
model = torch.compile(model)

max_steps = 50
optimizer = model.configure_optimizer(
    weight_decay=0.1,
    learning_rate=6e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    device=device
)
scheduler = GPT2CosineLearningRateScheduler(
    optimizer=optimizer,
    max_lr=6e-4,
    warmup_steps=10,
    max_steps=max_steps
)

for step, (x, y) in enumerate(data_loader):
    if step == max_steps: break
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    lr = scheduler.step(step)
    optimizer.step()
    print(f'step: {step:>3} | loss: {loss.item():8>.4f} | lr: {lr:8>.4e} | norm: {norm.item():8>.4f}')
