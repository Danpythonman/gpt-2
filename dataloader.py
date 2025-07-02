from pathlib import Path
import typing

import torch
import tiktoken


class DataLoader:

    _batch_size: int
    _block_size: int

    _tokens: torch.Tensor
    _current_position: int

    def __init__(self, batch_size: int, block_size: int, filepath: Path, device: torch.device = None):
        self._batch_size = batch_size
        self._block_size = block_size

        with open(filepath, 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self._tokens = torch.tensor(tokens, device=device)
        self._current_position = 0

        print(f'Loaded {len(tokens)} tokens')
        print(f'Each batch will have {batch_size * block_size} tokens')
        print(f'1 epoch = {len(self._tokens) // (batch_size * block_size)} batches')

    def __iter__(self):
        return self

    def __next__(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        B, T = self._batch_size, self._block_size

        start = self._current_position
        end = self._current_position + B * T + 1

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

        self._current_position += B * T

        if self._current_position + (B*T + 1) > len(self._tokens):
            self._current_position = 0

        return x, y
