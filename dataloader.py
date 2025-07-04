from logging_init import get_main_logger

import abc
import logging
from pathlib import Path
import typing

import numpy as np
import torch
import tiktoken


main_logger = get_main_logger()


class DataLoader(abc.ABC):
    '''
    Abstract base class for data loaders.

    Data loaders are iterators (hence the `__iter__` and `__next__` methods).
    They also implement the template method design pattern where they implement
    a method that runs after the next batch is calculated (`_post_batch_hook`)
    to perform extra steps specific to the data loader.
    '''

    _batch_size: int
    _block_size: int

    _process_rank: int
    _number_of_processes: int

    _device: typing.Optional[torch.device]
    _logger: logging.Logger

    _tokens: typing.Optional[torch.Tensor]

    def __init__(
        self,
        batch_size: int,
        block_size: int,
        process_rank: int,
        number_of_processes: int,
        device: typing.Optional[torch.device],
        logger: logging.Logger
    ):
        self._batch_size = batch_size
        self._block_size = block_size

        self._process_rank = process_rank
        self._number_of_processes = number_of_processes

        self._device = device
        self._logger = logger

    @abc.abstractmethod
    def _post_batch_hook(self) -> None:
        '''
        Template method that runs after the next batch is calculated.
        '''
        raise NotImplementedError()

    def __iter__(self):
        # No one should try to make this class an iterator (because it is an
        # abstract class; only its implementations should be made into
        # iterators)
        raise NotImplementedError()

    def __next__(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        B, T = self._batch_size, self._block_size

        start = self._current_position
        end = self._current_position + B * T + 1
        nproc = self._number_of_processes

        if self._tokens is None:
            raise Exception('self._tokens does not exist')

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

        # --- Template method ---
        self._post_batch_hook()

        return x, y


class ShakespeareDataLoader(DataLoader):
    '''
    Data loader for the Tiny Shakespeare dataset.
    '''

    _current_position: int

    def __init__(
        self,
        batch_size: int,
        block_size: int,
        filepath: Path,
        process_rank: int = 0,
        number_of_processes: int = 1,
        device: torch.device = None,
        logger: logging.Logger = main_logger
    ):
        super().__init__(
            batch_size=batch_size,
            block_size=block_size,
            process_rank=process_rank,
            number_of_processes=number_of_processes,
            device=device,
            logger=logger
        )

        with open(filepath, 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self._tokens = torch.tensor(tokens, device=device)

        self._current_position = batch_size * block_size * process_rank

        logger.info(f'Loaded {len(tokens)} tokens')
        logger.info(f'Each batch will have {batch_size * block_size} tokens')
        logger.info(f'1 epoch = {len(self._tokens) // (batch_size * block_size)} batches')

    def _post_batch_hook(self) -> None:
        # We don't need to do any extra steps after the next batch is calculated
        pass

    def __iter__(self):
        return self


class FineWebEdu10BDataLoader(DataLoader):
    '''
    Data loader for the FineWeb Edu 10B dataset.
    '''

    _shard_files: typing.List[Path]
    _current_shard: int
    _current_position: int

    def __init__(
        self,
        batch_size: int,
        block_size: int,
        split: typing.Union[typing.Literal['train'], typing.Literal['val']],
        process_rank: int = 0,
        number_of_processes: int = 1,
        device: typing.Optional[torch.device] = None,
        logger: logging.Logger = main_logger
    ):
        super().__init__(
            batch_size=batch_size,
            block_size=block_size,
            process_rank=process_rank,
            number_of_processes=number_of_processes,
            device=device,
            logger=logger
        )

        data_root = Path(__file__).parent / 'edu_fineweb10B'
        self._shard_files = sorted(list(file for file in data_root.iterdir() if split in file.name))

        if len(self._shard_files) == 0:
            raise Exception(f'No shards found for split {split}')

        logger.info(f'Found {len(self._shard_files)} shards for split {split}')

        self._current_shard = 0
        self._current_position = batch_size * block_size * process_rank
        self._tokens = self._load_tokens(self._shard_files[self._current_shard])

    def _load_tokens(self, filepath: Path) -> torch.Tensor:
        tokens_np = np.load(filepath)
        return torch.tensor(tokens_np, dtype=torch.long, device=self._device)

    def _post_batch_hook(self) -> None:
        B, T = self._batch_size, self._block_size
        nproc = self._number_of_processes

        # If we've reached the end of the current shard, go to the next shard
        if self._current_position + (B*T*nproc + 1) > len(self._tokens):
            self._current_shard = (self._current_shard + 1) % len(self._shard_files)
            self._tokens = self._load_tokens(self._shard_files[self._current_shard])

    def __iter__(self):
        return self
