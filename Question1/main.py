####################### Importing Libraries ################
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt 
from pprint import pprint
import random
import torch._dynamo
torch._dynamo.config.suppress_errors = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


block_size = 20 
emb_dim = 4
seed = 4000002
hidden_size = 100
model_path = "/Users/nimitt/Documents/ML/ML-ES335/assignment3/model_states/model.pth"


##################### Dataset Creating ######################
f = open("input.txt",'r')
text = f.read()
text = text.lower()
text = text.replace("\n","~")
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(text)))
stoi = {s:i for i,s in enumerate(chars)}
# stoi['~'] = 0
itos = {i:s for s,i in stoi.items()}
pprint(itos)





X, Y = [], []
context = []
for j in range(block_size):
  context = context + [stoi[text[j]]]

for i in range(block_size, len(text)):
    
  ch = text[i]
  ix = stoi[ch]
  X.append(context)
  Y.append(ix)
  print(''.join(itos[i] for i in context), '--->', itos[ix])
  context = context[1:] + [ix] # crop and append
  
# Move data to GPU

X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)

########################### Model ##############################
class NextChar(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = self.lin2(x)
    return x
  




model = NextChar(block_size, len(stoi), emb_dim, hidden_size).to(device)
model = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=0.01)


########################## training ############################

# Train the model

loss_fn = nn.CrossEntropyLoss()

import time
# Mini-batch training
batch_size = 10000
print_every = 100
elapsed_time = []
for epoch in range(10000):
    start_time = time.time()
    for i in range(0, X.shape[0], batch_size):
        x = X[i:i+batch_size]
        y = Y[i:i+batch_size]
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    end_time = time.time()
    elapsed_time.append(end_time - start_time)
    if epoch % print_every == 0:
        print(epoch, loss.item())


    if (epoch % 100 == 0):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': loss,
                    }, model_path)


########################## generate text ########################



g = torch.Generator()
g.manual_seed(seed)
def generate_text(model, prompt, itos, stoi, block_size, max_len=50):
    
    context = []
    for j in range(len(prompt)):
        context = context + [stoi[prompt[j]]]
    context = context[-block_size:]
        
    name = ''
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        # if ch == '~':
        #     break
        name += ch
        context = context[1:] + [ix]
    return name

######################## Testing ################################


# Load checkpoints

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
opt.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()

def print_text(text):
    for chr in text:
        if chr == '~':
            print("\n",end='')
        else:
            print(chr,end="")

generated_text = generate_text(model, itos, stoi, block_size, 100)
print_text(generated_text)