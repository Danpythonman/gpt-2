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

block_size = 1024
vocab_size=50_304
n_layer = 12
n_head = 12
n_embd = 768

total_batch_size = 2**19 # Approximately 0.5 million, as in the GPT-3 paper
mini_batches = 8
mini_batch_size = mini_batches * block_size

if total_batch_size % mini_batch_size != 0:
    raise Exception('Total batch size is not divisible by mini batch size')

grad_accum_steps = int(total_batch_size / mini_batch_size)

print(
    f'Total desired batch size is {total_batch_size:,} tokens, mini batch size '
    f'is {mini_batch_size:,} tokens ({mini_batches} examples of length '
    f'{block_size}), so gradients need to be accumulated for '
    f'{grad_accum_steps} mini batches per batch'
)

data_loader = DataLoader(
    batch_size=mini_batches,
    block_size=block_size,
    filepath=Path(__file__).parent / 'tiny-shakespeare.txt',
    device=device
)

model: GPT = GPT(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    vocab_size=vocab_size,
    flash_attention=True
)

print('Moving model to GPU')
model.to(device)

print('Compiling model')
model = torch.compile(model)

max_steps = 50

print('Configuring optimizer and learning rate scheduler')
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

print(f'Training for {max_steps} iterations')

for step, (x, y) in enumerate(data_loader):
    if step == max_steps: break

    optimizer.zero_grad()
    accumulated_loss = 0.0

    for micro_step in range(grad_accum_steps):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        loss /= grad_accum_steps
        accumulated_loss += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    lr = scheduler.step(step)
    optimizer.step()
    print(f'step: {step:>3} | loss: {accumulated_loss.item():9>.4f} | lr: {lr:8>.4e} | norm: {norm.item():8>.4f}')
