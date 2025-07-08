from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        '''
        `assert` 뒤에 오는 조건이 `False`면, 프로그램을 즉시 멈추고 `AssertionError`를 발생시킨다.
        이 경우, 전체 모델의 embedding dim 크기를 multi-head attention의 n_head 개수로 나눔으로써 
        0이 되는지 확인한다. 즉, 각 헤드에 균등하게 차원을 나누는지 확인하는 것이다.
        이는, 모델 세팅을 잘 했는지 확인하는 안전장치이다.
        '''
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is 'number of heads', hs is 'head size', and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs =64, so nh*hs=C=768 channels in the Transformer

        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal = True) # flash attention
        y = y.transpose(1,2).contiguous().view(B,T,C) # re-assemble all head outputs side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # 1. Multi-Head Attention 블록
        #   - self.ln_1(x)                : 먼저 LayerNorm으로 입력을 정규화하고 (안정성)
        #   - self.attn(self.ln_1(x))     : 그 결과를 Attention 연산에 통과시킨 후
        #   - x + self.attn(self.ln_1(x)) : 원래 입력 x를 더해줍니다 (정보 보존 및 기울기 전파)

        # 2. Feed-Forward Network (MLP) 블록
        #   - self.ln_2(x)                 : 위에서 나온 결과를 다시 LayerNorm으로 정규화하고 (안정성)
        #   - self.mlp(self.ln_2(x))       : 그 결과를 MLP 연산에 통과시킨 후
        #   - x + self.mlp(self.ln_2(x))   : 다시 원래 입력(정확히는 Attention 블록의 출력)을 더해줍니다.

        x = x + self.attn(self.ln_1(x))  # 1.
        x = x + self.mlp(self.ln_2(x))   # 2.
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)


