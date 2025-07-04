# %% In a series of stages we will optimize the training of GPT-2 so that it
# takes less time.

from pathlib import Path
import time
import typing

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import torch
from torch.nn import functional as F

from dataloader import ShakespeareDataLoader
from gpt2 import GPT


if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Using device: {device}')


def print_title(title: str) -> None:
    print()
    print(title)
    print('=' * len(title))


def print_diagnostics(loss: torch.types.Number, t0: float, t1: float, data_loader: ShakespeareDataLoader) -> None:
    dt = t1 - t0
    tokens_per_sec = int(data_loader._batch_size * data_loader._block_size // dt)
    print(f'step {i:>3}, loss: {loss:>8.4f}, dt: {dt*1000:<8.4f}ms, tokens/sec: {tokens_per_sec:<8,d}')


times: typing.Dict[str, typing.List[float]] = dict()

# %% In the first stage we show the default performance without any optimization
# strategies applied.

stage = 'default'
print_title(stage)
times[stage] = []

batch_size = 8
block_size = 1024
vocab_size=50_257
n_layer = 8
n_head = 8
n_embd = 768

data_loader = ShakespeareDataLoader(
    batch_size=batch_size,
    block_size=block_size,
    filepath=Path(__file__).parent / 'tiny-shakespeare.txt',
    device=device
)

model = GPT(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_head,
    block_size=block_size,
    vocab_size=vocab_size,
    flash_attention=False
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i, (x, y) in enumerate(data_loader):
    if i == 50: break
    t0 = time.time()
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    print_diagnostics(loss.item(), t0, t1, data_loader)
    times[stage].append(t1-t0)

# %% In the second stage we set PyTorch's matrix multiplication precision to
# 'high' instead of 'highest'. This makes the values in matrices take up less
# memory, allowing more operations to be performed on the GPU with less memory
# and less moving tensors between the GPU caches and the GPU's main memory.

stage = 'lower precision'
print_title(stage)
times[stage] = []

torch.set_float32_matmul_precision('high')

batch_size = 8
block_size = 1024
vocab_size=50_257
n_layer = 8
n_head = 8
n_embd = 768

data_loader = ShakespeareDataLoader(
    batch_size=batch_size,
    block_size=block_size,
    filepath=Path(__file__).parent / 'tiny-shakespeare.txt',
    device=device
)

model = GPT(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_head,
    block_size=block_size,
    vocab_size=vocab_size,
    flash_attention=False
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i, (x, y) in enumerate(data_loader):
    if i == 50: break
    t0 = time.time()
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    print_diagnostics(loss.item(), t0, t1, data_loader)
    times[stage].append(t1-t0)
# %% The third stage is similar to the second stage, but more aggressive. We use
# PyTorch autocasting to further reduce the precision of matrix values to
# BFloat16, further increasing the amount of tensors we can store in the GPU
# caches.

stage = 'autocast bfloat16'
print_title(stage)
times[stage] = []

torch.set_float32_matmul_precision('high')

batch_size = 8
block_size = 1024
vocab_size=50_257
n_layer = 8
n_head = 8
n_embd = 768

data_loader = ShakespeareDataLoader(
    batch_size=batch_size,
    block_size=block_size,
    filepath=Path(__file__).parent / 'tiny-shakespeare.txt',
    device=device
)

model = GPT(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_head,
    block_size=block_size,
    vocab_size=vocab_size,
    flash_attention=False
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i, (x, y) in enumerate(data_loader):
    if i == 50: break
    t0 = time.time()
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    print_diagnostics(loss.item(), t0, t1, data_loader)
    times[stage].append(t1-t0)

# %% For the next optimization, we compile the model, removing the overhead of
# the Python interpreter. Instead of executing the forward pass of the model
# line-by-line, as the Python interpreter would, the model is compiled to
# execute efficiently on the GPU without interference from the interpreter.

stage = 'compiled model'
print_title(stage)
times[stage] = []

torch.set_float32_matmul_precision('high')

batch_size = 8
block_size = 1024
vocab_size=50_257
n_layer = 8
n_head = 8
n_embd = 768

data_loader = ShakespeareDataLoader(
    batch_size=batch_size,
    block_size=block_size,
    filepath=Path(__file__).parent / 'tiny-shakespeare.txt',
    device=device
)

model = GPT(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_head,
    block_size=block_size,
    vocab_size=vocab_size,
    flash_attention=False
)
model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i, (x, y) in enumerate(data_loader):
    if i == 50: break
    t0 = time.time()
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    print_diagnostics(loss.item(), t0, t1, data_loader)
    times[stage].append(t1-t0)

# %% Next, we use the flash attention calculation
# (F.scaled_dot_product_attention(q, k, v, is_causal=True)) instead of the
# manual matrix multiplications and softmax calculations.

stage = 'flash attention'
print_title(stage)
times[stage] = []

torch.set_float32_matmul_precision('high')

batch_size = 8
block_size = 1024
vocab_size=50_257
n_layer = 8
n_head = 8
n_embd = 768

data_loader = ShakespeareDataLoader(
    batch_size=batch_size,
    block_size=block_size,
    filepath=Path(__file__).parent / 'tiny-shakespeare.txt',
    device=device
)

model = GPT(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_head,
    block_size=block_size,
    vocab_size=vocab_size,
    flash_attention=True
)
model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i, (x, y) in enumerate(data_loader):
    if i == 50: break
    t0 = time.time()
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_per_sec = data_loader._batch_size * data_loader._block_size / dt
    print_diagnostics(loss.item(), t0, t1, data_loader)
    times[stage].append(t1-t0)

# %% Finally, we set vocab size to a value that can be divided by 2 many times.
# This helps performance one the GPU because it doesn't have to deal with
# special cases of odd number dimensions.

stage = 'vocab size power of 2'
print_title(stage)
times[stage] = []

torch.set_float32_matmul_precision('high')

batch_size = 8
block_size = 1024
vocab_size=50_304
n_layer = 8
n_head = 8
n_embd = 768

data_loader = ShakespeareDataLoader(
    batch_size=batch_size,
    block_size=block_size,
    filepath=Path(__file__).parent / 'tiny-shakespeare.txt',
    device=device
)

model = GPT(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_head,
    block_size=block_size,
    vocab_size=vocab_size,
    flash_attention=True
)
model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i, (x, y) in enumerate(data_loader):
    if i == 50: break
    t0 = time.time()
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    print_diagnostics(loss.item(), t0, t1, data_loader)
    times[stage].append(t1-t0)

# %% Plot histograms of the optimization performances.

max_value = max(t for ts in times.values() for t in ts)
plt.figure()
for label, data in times.items():
    kde = gaussian_kde(data)
    x_range = np.linspace(0, max_value, 1000)
    y = kde(x_range)
    plt.plot(x_range, y, label=label)
    plt.fill_between(x_range, y, alpha=0.3)
plt.title('Optimization Strategies Performance')
plt.xlabel('Time per training iteration (seconds)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
