from logging_init import get_main_logger

import logging
from pathlib import Path
import typing

import torch
import tiktoken


main_logger = get_main_logger()


class DataLoader:

    _batch_size: int
    _block_size: int

    _process_rank: int
    _number_of_processes: int

    _tokens: torch.Tensor
    _current_position: int

    def __init__(
        self,
        batch_size: int,
        block_size: int,
        filepath: Path,
        process_rank: int,
        number_of_processes: int,
        device: torch.device = None,
        logger: logging.Logger = main_logger
    ):
        self._batch_size = batch_size
        self._block_size = block_size

        self._process_rank = process_rank
        self._number_of_processes = number_of_processes

        with open(filepath, 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self._tokens = torch.tensor(tokens, device=device)

        self._current_position = batch_size * block_size * process_rank

        logger.info(f'Loaded {len(tokens)} tokens')
        logger.info(f'Each batch will have {batch_size * block_size} tokens')
        logger.info(f'1 epoch = {len(self._tokens) // (batch_size * block_size)} batches')

    def __iter__(self):
        return self

    def __next__(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        B, T = self._batch_size, self._block_size

        start = self._current_position
        end = self._current_position + B * T + 1
        nproc = self._number_of_processes

        buf = self._tokens[start:end] # (B*T+1,)

        # Sequence of x-values at up to index i is a sequence of tokens, and the
        # y-value at index i is the next token in that sequence.
        #
        # We ignore the last token in the sequence for the x-values (because
        # there is no corresponding y-value at the end of the sequence) and we
        # ignore the first token for the y-values (because there is no
        # corresponding sequence of x-values to predict from).
        x = buf[:-1].view(B, T) # (B, T)
        y = buf[1:].view(B, T)  # (B, T)

        self._current_position += B * T * nproc

        if self._current_position + (B*T*nproc + 1) > len(self._tokens):
            self._current_position = B * T * nproc

        return x, y
