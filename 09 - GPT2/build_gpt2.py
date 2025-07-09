import math
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
        이 경우, 각 헤드에 균등하게 차원을 나누는지 확인하는 안전 장치와 같다.
        '''
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer('bias',
                             torch.trill(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size, config.block_size))


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is 'number of heads', hs is 'head size', and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs =64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)

        # k                     : (B, T, C)
        # k.view                : (B, T, nh, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        # 이제 Pytorch는 첫 번째 차원(B, nh)을 배치 차원으로 인식하고,
        # 나머지 두 차원(T, hs)에 대해 행렬곱을 수행한다.
        # attention (materializers the large (T,T) matrix for all the queries and keys)

        # q                   : (B, nh, T, hs)
        # k.transpose(-2, -1) : (B, nh, hs, T)
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T,  T)

        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

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
        #   - x + self.attn(self.ln_1(x)) : 원래 입력 x를 더해줍니다 (정보 보존 및 기울기 전파) Residual connection

        # 2. Feed-Forward Network (MLP) 블록
        #   - self.ln_2(x)                 : 위에서 나온 결과를 다시 LayerNorm으로 정규화하고 (안정성)
        #   - self.mlp(self.ln_2(x))       : 그 결과를 MLP 연산에 통과시킨 후
        #   - x + self.mlp(self.ln_2(x))   : 다시 원래 입력(정확히는 Attention 블록의 출력)을 더해줍니다. Residual connection

        x = x + self.attn(self.ln_1(x))  # 1.에 해당
        x = x + self.mlp(self.ln_2(x))   # 2.에 해당
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
            wte = nn.Embedding(config.vocab_size, config.n_embd),               # tokenized vector
            wpe = nn.Embedding(config.block_size, config.n_embd),               # position vector
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # hidden layer
            ln_f = nn.LayerNorm(config.n_embd),                                 # layer normalize
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)


