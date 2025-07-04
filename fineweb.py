import multiprocessing as mp
import os
from pathlib import Path

from datasets import load_dataset
import numpy as np
import tiktoken
from tqdm import tqdm

dataset_dir = Path(__file__).parent / 'edu_fineweb10B'
dataset_dir.mkdir(exist_ok=True)

remote_name = 'sample-10BT'

# 10 billion tokens total, 100 million tokens per shard. so 100 shards total
shard_size = 100_000_000

fw = load_dataset('HuggingFaceFW/fineweb-edu', name=remote_name, split='train')

enc = tiktoken.get_encoding('gpt2')

eot = enc._special_tokens['<|endoftext|>']


def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    if (tokens_np < 0).any() or (tokens_np >= 2**16).any():
        raise Exception('token dictionary too large for uint16')
    return tokens_np.astype(np.uint16)


def generate_datafile_path(shard_index: int) -> Path:
    split = 'val' if shard_index == 0 else 'train'
    filename = f'edufineweb_{split}_{shard_index:02d}'
    return dataset_dir / filename


def write_datafile(filename: str, tokens_np: np.ndarray):
    np.save(filename, tokens_np)


nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # Check if this chunk of tokens can fit in the file we are making
        if token_count + len(tokens) < shard_size:
            # If this chunk of tokens fits, put it in the NumPy array
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'Shard {shard_index}')
            progress_bar.update(len(tokens))
        else:
            # If this chunk of tokens does not fit, put whatever we can into the
            # NumPy array
            filepath = generate_datafile_path(shard_index)
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            # Saved the filled-up NumPy array to a file
            write_datafile(filepath, all_tokens_np)
            # Put the remaining tokens that didn't fit into the beginning of the
            # NumPy array for the next iteration
            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # Write remaining tokens to the last shard
    if token_count != 0:
        filepath = generate_datafile_path(shard_index)
        write_datafile(filepath, all_tokens_np[:token_count])
