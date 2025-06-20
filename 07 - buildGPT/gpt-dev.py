import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters

batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

# ------------

torch.manual_seed(1337)

# wget 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
# read it in to inspect it
with open('input.txt',mode = 'r',encoding='utf-8') as f:
    text = f.read()


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder : take a list of intergers, output a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
# Let's now split up the data into trainn and validation sets
n = int(0.9 * len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    # data는 여기서 불러온다.
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # (high, size)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # return (batch_size) x (block_size) tensor = 4 * 8
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """
    Estimate the loss of the model on the train and val splits.

    This function calculates the average loss over a fixed number of iterations
    (`eval_iters`). By averaging the losses from multiple batches, it reduces
    the noise from any single batch evaluation, providing a more stable and
    statistically reliable measure of the model's performance.
    """
    out = {}
    model.eval()                            # Set the module in evaluation mode / inherit from 'nn.Module'
    for split in ['train', 'val']:          # 'train', 'val'로 구분
        losses = torch.zeros(eval_iters)    # eval_iters횟수만큼 loss 값을 저장할 텐서 생성
        for k in range(eval_iters):
            X, Y = get_batch(split)         # 배치 데이터 가져오기
            logits, loss = model(X, Y)
            losses[k] = loss.item()         # 손실값을 python 숫자로 변환하여 저장
        out[split] = losses.mean()          # 'out' will be {'train" : num, 'val': num}
    model.train()                           # Set the module in training mode / inherit from 'nn.Module'
    return out


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)
        # B : batch size
        # T : Time -> block size
        # C : channel, at this situations, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # 2 dimensional
            targets = targets.view(B * T)  # 1 dimensional
            loss = F.cross_entropy(logits, targets)  # negative log likelihood
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            # self(idx)는 뭘까?
            # nn.Module을 상속받았기에, 자동으로 __call__ 메서드를 갖으며,
            # forward 메서드로 연결된다.
            # 즉, self(idx) == self.forward(idx)
            logits, loss = self(idx)
            # focus only on the last time step,
            # we pluck out last element at the time dimension
            logits = logits[:, -1, :]  # becomes (B,C)
            # apply softmax to get probabilities (vocab들의 분포 확률을 구한다.)
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution (distribution에서 뽑는다.)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence (뽑은 결과를 idx에 새로 추가한다.)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
model = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):

    # every onec in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss: {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
