import numpy as np
import torch
from torch import nn
import math
class Embedding(nn.Module):
    def __init__(self , vocab_size , d_model):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size , d_model)

    def forward(self , x):
        return self.embedding(x)
    
class PossitionalEncoding(nn.Module):
    def __init__(self , d_model , max_len = 5000):
        super().__init__()

        pe = torch.zeros(max_len , d_model)

        positions = torch.arange(0 , max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))

        pe[:,0::2] = torch.sin(positions*div_term)
        pe[:,1::2] = torch.cos(positions*div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe" , pe)

    def forward(self , x):
        max_len = x.size(1)

        return x + self.pe[:,:max_len]
    
if __name__ == "__main__":
    
    vocab = {
    "i": 0,
    "love": 1,
    "deep": 2,
    "learning": 3
    }
    vocab_size = len(vocab)
    d_model = 8
    embedd = Embedding(vocab_size , d_model)

    sentence = [i for i in vocab] # sentence =['i', 'love', 'deep', 'learning']
    
    token_id = torch.tensor([
        [vocab[words] for words in sentence]  # tensor([0, 1, 2, 3]) -> ('i', 'love', 'deep', 'learning')
    ])

    Embeddings = embedd(token_id) 
    
    print("embedding without the positional encoding: ")
    print()
    
    for i, word in enumerate(sentence):
        print(f"{word} = {Embeddings[0, i]}")

    #Embedding with positional encoding

    positional_encod = PossitionalEncoding(d_model , max_len=100)
    PosEncoded = positional_encod(Embeddings)

    print("embedding with the positional encoding: ")
    print()
    
    for i, word in enumerate(sentence):
        print(f"{word} = {PosEncoded[0, i]}")


