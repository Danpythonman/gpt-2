import abc
import math

import torch


class LearningRateScheduler(abc.ABC):

    def get_lr(self, iteration: int) -> float:
        '''
        Calculates the learning rate based on iteration number.
        '''
        raise NotImplementedError

    def step(self, iteration: int) -> float:
        '''
        Calculates the next learning rate, applies it to the optimizer, and
        returns the learning rate.
        '''
        raise NotImplementedError


class GPT2CosineLearningRateScheduler(LearningRateScheduler):

    optimizer: torch.optim.Optimizer
    max_lr: float
    min_lr: float
    warmup_steps: int
    max_steps: int

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float = 3e-4,
        warmup_steps: int = 10,
        max_steps: int = 50
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = max_lr * 0.1
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, iteration: int) -> float:
        if iteration < self.warmup_steps:
            return self.max_lr * (iteration + 1) / self.warmup_steps

        if iteration > self.max_steps:
            return self.min_lr

        decay_ratio = (iteration - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        if decay_ratio < 0:
            raise Exception('decay ratio is less than 0')
        if decay_ratio > 1:
            raise Exception('decay ratio is greater than 1')

        coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coefficient * (self.max_lr - self.min_lr)

    def step(self, iteration: int) -> float:
        lr = self.get_lr(iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
