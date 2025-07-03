from logging_init import get_main_logger, get_worker_logger, get_null_logger

import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloader import DataLoader
from gpt2 import GPT
from lr_scheduler import GPT2CosineLearningRateScheduler


using_ddp = int(os.environ.get('RANK', -1)) != -1
if using_ddp:
    if not torch.cuda.is_available():
        raise Exception('Need CUDA for DDP')
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    main_logger = get_main_logger() if ddp_rank == 0 else get_null_logger()
    worker_logger = get_worker_logger(ddp_rank)

    main_logger.info('Using DDP')
    worker_logger.info(f'Worker {ddp_rank} using device: {device}')
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    main_logger = get_main_logger()
    worker_logger = main_logger

    main_logger.info('Not using DDP')
    main_logger.info(f'Using device: {device}')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

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

if grad_accum_steps % ddp_world_size != 0:
    raise Exception('Total batch size is not divisible by number of GPUs')

grad_accum_steps_per_gpu = grad_accum_steps / ddp_world_size

main_logger.info(
    f'Total desired batch size is {total_batch_size:,} tokens, mini batch size '
    f'is {mini_batch_size:,} tokens ({mini_batches} examples of length '
    f'{block_size}), so gradients need to be accumulated for '
    f'{grad_accum_steps} mini batches per batch. Since we have '
    f'{ddp_world_size} GPUs, each GPU will have a batch size of '
    f'{grad_accum_steps_per_gpu}.'
)

data_loader = DataLoader(
    batch_size=mini_batches,
    block_size=block_size,
    process_rank=ddp_rank,
    number_of_processes=ddp_world_size,
    filepath=Path(__file__).parent / 'tiny-shakespeare.txt',
    device=device,
    logger=main_logger
)

model: GPT = GPT(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    vocab_size=vocab_size,
    flash_attention=True,
    logger=main_logger
)

worker_logger.info('Moving model to GPU')
model.to(device)

worker_logger.info('Compiling model')
model = torch.compile(model)

if using_ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

if isinstance(model, DDP):
    raw_model = model.module
else:
    raw_model = model

max_steps = 50

worker_logger.info('Configuring optimizer and learning rate scheduler')
optimizer = raw_model.configure_optimizer(
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

worker_logger.info(f'Training for {max_steps} iterations')

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
        if using_ddp and isinstance(model, DDP):
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()

    if using_ddp:
        torch.distributed.all_reduce(accumulated_loss, op=torch.distributed.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    lr = scheduler.step(step)
    optimizer.step()
    main_logger.info(f'step: {step:>3} | loss: {accumulated_loss.item():9>.4f} | lr: {lr:8>.4e} | norm: {norm.item():8>.4f}')

if using_ddp:
    destroy_process_group()
