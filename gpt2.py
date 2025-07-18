from logging_init import get_main_logger

import abc
import inspect
import logging
import math
import typing

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


main_logger = get_main_logger()


class CausalSelfAttention(nn.Module, abc.ABC):
    '''
    Causal multi-head self-attention.
    '''

    n_head: int
    '''
    Number of attention heads.
    '''

    n_embd: int
    '''
    Embedding size of the attention heads.
    '''

    block_size: int
    '''
    Context length (number of tokens that attention is applied to).
    '''

    c_attn: nn.Linear
    '''
    Concatenated attention matrix containing the key, query, and value
    projections for all the attention heads.

    Shape of the matrix should be `(n_embd, n_embd * 3)` because the input
    vector of size `n_embd` maps to three vectors of size `n_embd` (key, query,
    and values vectors).
    '''

    c_proj: nn.Linear
    '''
    Output projection layer.

    Shape of the matrix should be of size `(n_embd, n_embd)`.
    '''

    bias: Tensor
    '''
    Triangular matrix to enable auto regression (makes attention only apply to
    past tokens, never future tokens).

    Should be of shape `(1, 1, block_size, block_size)`. The first `1` is the
    batch dimension and the second `1` is the head dimension. (We used `1`
    instead of batch size or number of heads because the mask is the same for
    every batch and head, so we just let PyTorch broadcast it for us.)
    '''

    def __init__(self, n_head: int, n_embd: int, block_size: int):
        super().__init__()

        if n_embd % n_head != 0:
            raise Exception('n_head must be a factor of n_embd')

        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

        self.c_attn = nn.Linear(n_embd, n_embd * 3)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(block_size, block_size))
                .unsqueeze(0)
                .unsqueeze(0)
        )

        self.c_proj.INIT_SCALE_DOWN = True

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class ManualCausalSelfAttention(CausalSelfAttention):
    '''
    Causal multi-head self-attention with manual attention calculations.
    '''

    def forward(self, x: Tensor) -> Tensor:
        # Shape is `(batch, time, channel)` which corresponds to
        # `(batch, block_size, n_embd)`
        B, T, C = x.shape

        qkv: Tensor = self.c_attn(x) # (B, T, C) -> (B, T, C*3)

        # Splitting across the second dimension of the `(B, T, C*3)` tensor with
        # split size `n_embd = C` yields three tensors of size `(B, T, C)`
        q: Tensor
        k: Tensor
        v: Tensor
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Split the channel dimension into multiple heads
        nh = self.n_head
        hs = C // nh
        k = k.view(B, T, nh, hs).transpose(1, 2) # (B, T, C) -> (B, nh, T, hs)
        q = q.view(B, T, nh, hs).transpose(1, 2) # (B, T, C) -> (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2) # (B, T, C) -> (B, nh, T, hs)

        # Attention calculation:
        #
        #    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        #
        # Attention matrix will have shape `(B, nh, T ,T)`
        d_k = k.shape[-1]
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(d_k))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, nh, T, hs) -> (B, T, C)

        y = self.c_proj(y) # (B, T, C) -> (B, T, C)

        return y


class FlashCausalSelfAttention(CausalSelfAttention):
    '''
    Causal multi-head self-attention using flash attention calculation via
    PyTorch's function:

        torch.n.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    '''

    def forward(self, x: Tensor) -> Tensor:
        # Shape is `(batch, time, channel)` which corresponds to
        # `(batch, block_size, n_embd)`
        B, T, C = x.shape

        qkv: Tensor = self.c_attn(x) # (B, T, C) -> (B, T, C*3)

        # Splitting across the second dimension of the `(B, T, C*3)` tensor with
        # split size `n_embd = C` yields three tensors of size `(B, T, C)`
        q: Tensor
        k: Tensor
        v: Tensor
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Split the channel dimension into multiple heads
        nh = self.n_head
        hs = C // nh
        k = k.view(B, T, nh, hs).transpose(1, 2) # (B, T, C) -> (B, nh, T, hs)
        q = q.view(B, T, nh, hs).transpose(1, 2) # (B, T, C) -> (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2) # (B, T, C) -> (B, nh, T, hs)

        # Flash attention calculation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, nh, T, hs) -> (B, T, C)

        y = self.c_proj(y) # (B, T, C) -> (B, T, C)

        return y


class MLP(nn.Module):
    '''
    Multi-layer perceptron.
    '''

    n_embd: int
    '''
    Embedding size.
    '''

    c_fc: nn.Linear
    '''
    Fully connected layer.
    '''

    gelu: nn.GELU
    '''
    GELU non-linearity layer.
    '''

    c_proj: nn.Linear
    '''
    Projection layer.
    '''

    def __init__(self, n_embd: int):
        super().__init__()

        self.n_embd = n_embd

        self.c_fc = nn.Linear(n_embd, n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(n_embd * 4, n_embd)

        self.c_proj.INIT_SCALE_DOWN = True

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    '''
    Transformer block.
    '''

    n_embd: int
    '''
    Embedding size.
    '''

    ln_1: nn.LayerNorm
    '''
    First layer normalization layer before self attention is applied.
    '''

    attn: CausalSelfAttention
    '''
    Self attention layer.
    '''

    ln_2: nn.LayerNorm
    '''
    Second layer normalization layer after self attention is applied.
    '''

    mlp: MLP
    '''
    Fully connected layer after the second layer normalization is applied.
    '''

    def __init__(self, n_head: int, n_embd: int, block_size: int, flash_attention: bool = True):
        super().__init__()

        self.n_embd = n_embd

        self.ln_1 = nn.LayerNorm(n_embd)
        if flash_attention:
            self.attn = FlashCausalSelfAttention(n_head, n_embd, block_size)
        else:
            self.attn = ManualCausalSelfAttention(n_head, n_embd, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int

    transformer: nn.ModuleDict
    '''
    Transformer, containing token embeddings, position embeddings, hidden
    attention layers, and a final layer normalization layer.
    '''

    lm_head: nn.Linear
    '''
    Language model head.
    '''

    _logger: logging.Logger

    def __init__(
        self,
        n_layer: int,
        n_head: int,
        n_embd: int,
        block_size: int,
        vocab_size: int,
        flash_attention: bool = True,
        logger: logging.Logger = main_logger
    ):
        super().__init__()

        self._logger = logger

        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            h = nn.ModuleList([Block(n_head, n_embd, block_size, flash_attention) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd)
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Parameter sharing
        self.transformer.wte.weight = self.lm_head.weight
        self.transformer.wpe.WPE_INIT = True

        # Initialize weights of this module's parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        '''
        Initialization of parameters according to OpenAI's GPT-2 implementation.

        Most of the weight matrices are initialized to a normal distribution
        with a mean of 0.0 and a standard deviation of 0.02. The bias vectors
        are initialized to 0. The position embedding matrix differs in that it
        is initialized with a standard deviation of 0.01.

        Additionally residual connections have their weights scaled down by
        `1/sqrt(n_layer)` to keep the standard deviation close to 1.

        See https://github.com/openai/gpt-2/blob/master/src/model.py
        '''

        if isinstance(module, nn.Linear):
            if hasattr(module, 'INIT_SCALE_DOWN'):
                # `n_layer` is multiplied by 2 because there are two residual
                # connections in each `Block`
                std = 0.02 * (2 * self.n_layer) ** -0.5
            else:
                std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if hasattr(module, 'WPE_INIT'):
                std = 0.01
            else:
                std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def configure_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: typing.Tuple[float, float],
        eps: float,
        device: torch.device
    ) -> torch.optim.Optimizer:
        '''
        Sets up the AdamW optimizer with weight decay on the weight matrices.
        '''

        param_dict = {
            name: parameter
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
        }

        # Weight matrices will be weight-decayed, bias vectors will be not
        decay_params, non_decay_params = [], []
        for parameter in param_dict.values():
            if parameter.dim() >= 2:
                decay_params.append(parameter)
            else:
                non_decay_params.append(parameter)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': non_decay_params, 'weight_decay': 0.0}
        ]

        self._logger.info(
            f'Decayed parameter tensors: {len(decay_params)} '
            f'({sum(p.numel() for p in decay_params):,} total decayed '
            f'parameters)\nNon-decayed parameter tensors: '
            f'{len(non_decay_params)} '
            f'({sum(p.numel() for p in non_decay_params):,} total non-decayed '
            f'parameters'
        )

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        using_cuda = 'cuda' in device
        use_fused = fused_available and using_cuda
        self._logger.info(f'Using fused: {use_fused}')

        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            fused=use_fused
        )

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape

        if T > self.block_size:
            raise Exception('Sequence length is larger than block size')

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T,)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)

        x = tok_emb + pos_emb # (B, T, n_embd)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) # (B, T, vocab_size)

        return logits

    @classmethod
    def default(cls) -> 'GPT':
        return GPT(
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=1024,
            vocab_size=50_257
        )

    @classmethod
    def from_pretrained(cls, model_type: str) -> 'GPT':
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)
        }

        if model_type not in config_args.keys():
            raise Exception('Invalid model type')

        print(f'Loading weights from pretrained GPT {model_type}')

        config = config_args[model_type]
        vocab_size = 50_257 # 256 byte tokens + 50,000 BPE merges + 1 <|endoftext|> token
        block_size = 1024

        model = GPT(
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_embd=config['n_embd'],
            block_size=block_size,
            vocab_size=vocab_size
        )

        sd = model.state_dict()
        # We exclude the attention bias because is is a buffer that we
        # initialize manually in the `CausalSelfAttention` class
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.bias') and not k.endswith('.attn.masked_bias')]

        # These parameters need to be transposed to work with our implementation
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        if len(sd_keys) != len(sd_keys_hf):
            raise Exception('State dict and Hugging Face state dict have different keys')

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                if sd_hf[k].shape[::-1] != sd[k].shape:
                    raise Exception(f'Transposed key shape mismatch: {k}')
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                if sd_hf[k].shape != sd[k].shape:
                    raise Exception(f'Key shape mismatch: {k}')
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
