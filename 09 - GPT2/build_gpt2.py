import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# 2:15:08

# ----------------------------------------------------------------------------------------------------------------------
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
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer('bias',
                             torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size, config.block_size))


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

        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T,  T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        ## flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

# ----------------------------------------------------------------------------------------------------------------------
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

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

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
        '''torch.nn.Module.apply 
        : Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.
        Typical use includes initializing the parameters of a model.
        '''

    def _init_weights(self, module):
        '''
        가중치를 초기화하는 method
        '''
        if isinstance(module, nn.Linear):
            std = 0.02
            # residual connection으로 인해 학습 초기의 불안정이나 exploding되는 문제 방지
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std=0.02)

    def forward(self, idx, targets = None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f'Cannot forward sequence of length {T}, block size is only {self.config.block_size}'
        # forward the token and position embeddings
        # 07-buildGPT의 'gpt-dev' 참고
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        # 마지막 Linear 레이어, 각 토큰 위치에서 다음에 올 토큰이 무엇일지 예측하는 단계
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print('loading weights from pretrained gpt: %s' % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2'        : dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium' : dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large'  : dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl'     : dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params

        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        # .state_dict() : 모델의 각 layer의 parameter(가중치와 편향등)가 매핑된 dictionary.
                        # Pytorch로 모델을 저장하거나 불러오고 싶을 때 사용한다.
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        # huggingface 모델의 각 key에 대해서 다음과 같이 반복한다.
        for k in sd_keys_hf:
            # 현재 키가 전치가 필요한 가중치인지 확인한다
            # any : 인자로 넘어온 자료구조 내의 하나의 요소라도 참이면 True, 모두 거짓이라면 False
                # key가 transposed에 속한 것으로 endswith한다면,
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                # sd_hf[k]를 전치했을 때의 shape와 sd[k]의 shape가 같은지 확인하고, 아니라면 AssertError 뱉는다.
                assert sd_hf[k].shape[::-1] == sd[k].shape
                # gradient 계산을 비활성화한 상태에서
                with torch.no_grad():
                    # sd_hf[k].t() : 전치한 상태
                    # sd[k]에 위의 값을 복사한다.
                    sd[k].copy_(sd_hf[k].t())
            # 전치가 필요없다면, 단순 그대로 복사한다.
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# ---------------------------------------------------------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('..\\data\\input.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# ---------------------------------------------------------------------------------------------------------------------
# attempt to autodetect the device
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

num_return_sequences = 5
max_length = 30

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# get a data batch
train_loader = DataLoaderLite(4, 1024)

# Sets the internal precision of float32 matrix multiplications
torch.set_float32_matmul_precision('high')

# get logits
# override vocab_size number, make it much prettier
model = GPT(GPTConfig(vocab_size = 50304))
model.to(device)
model = torch.compile(model)

# optimize
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # Enables autocasting for the forward pass
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # Forward pass with BF16
        logits, loss = model(x, y)
        # import code; code.interact(local=locals())
    # Backward pass with Loss Scaling and BF16, and convert into FP32
    loss.backward()
    # update the parameter to derease the loss, FP32
    optimizer.step()

    ''' 
    GPU의 비동기 연산 때문에 필요한 코드, 
    CPU가 GPU에서 진행 중인 모든 작업이 완료될 때까지 기다리도록 만드는 명령어이다.
    '''
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f'step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}')

import sys; sys.exit(0)














