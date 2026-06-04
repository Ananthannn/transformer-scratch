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

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self , Q , K , V , mask = None):
        d_k = Q.size(-1)

        score = (Q@K.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            score = score.masked_fill(
                mask == 0,
                -1e9
            )
        
        attention = torch.softmax(
            score,
            dim=-1
        )

        output = attention@V

        return output , attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model , num_head):
        super().__init__()

        self.d_model = d_model
        self.num_head = num_head

        self.d_k = (d_model // num_head)

        self.Wq = nn.Linear(d_model , d_model)
        self.Wk = nn.Linear(d_model , d_model)
        self.Wv = nn.Linear(d_model , d_model) 

        self.fc = nn.Linear(d_model , d_model)

        self.attention = (
            ScaledDotProductAttention()
        )

    def forward(self , q , k , v , mask = None):
        
        batch_size = q.size(0)

        Q = self.Wq(q)
        K = self.Wk(k)
        V = self.Wv(v)

        Q = Q.view(
            batch_size,
            -1,
            self.num_head,
            self.d_k
        ).transpose(1,2)
        
        K = K.view(
            batch_size,
            -1,
            self.num_head,
            self.d_k
        ).transpose(1,2)

        V = V.view(
            batch_size,
            -1,
            self.num_head,
            self.d_k
        ).transpose(1,2)

        output , _ = self.attention(Q , K , V , mask)

        output = output.transpose(1,2).contiguous()

        output = output.view(batch_size , -1 , self.d_model)

        return self.fc(output)

class FeedForwardNetwork(nn.Module):
    def __init__(self , d_model , num_neuron):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model , num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron , d_model)
        )

    def forward(self , x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self , d_model ,num_head , num_neuron):
        super().__init__()

        self.MHA = MultiHeadAttention(d_model , num_head)

        self.FFN = FeedForwardNetwork(d_model , num_neuron)

        self.AddAndNorm1 = nn.LayerNorm(d_model)

        self.AddAndNorm2 = nn.LayerNorm(d_model)

    def forward(self , x , mask = None):
        
        AttnOut = self.MHA(x,x,x,mask)
        
        x = self.AddAndNorm1(x+AttnOut)

        FFNOut = self.FFN(x)

        x = self.AddAndNorm2(x + FFNOut)

        return x
    
class Encoder(nn.Module):
    def __init__(self , vocab_size , d_model , num_heads , num_neurons , num_encoders):
        super().__init__()

        self.embedd = Embedding(vocab_size , d_model)
        self.positionEmbedd = PossitionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model , num_heads , num_neurons) for _ in range(num_encoders)
            ]
        )
    
    def forward(self , x , mask = None):
        
        # embedding the content in the corpus
        x = self.embedd(x)

        # adding the positional encoding to the embeddings
        x = self.positionEmbedd(x)

        for layer in self.layers:
            x = layer(x , mask)

        return x

class DecoderLayer(nn.Module):
    def __init__(self , d_model , num_heads , num_neurons):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model , num_heads)

        self.cross_attn = MultiHeadAttention(d_model , num_heads)

        self.FFN = FeedForwardNetwork(d_model , num_neurons)

        self.AddAndNorm1 = nn.LayerNorm(d_model)

        self.AddAndNorm2 = nn.LayerNorm(d_model)

        self.AddAndNorm3 = nn.LayerNorm(d_model)

    def forward(self , x , encoder_output , src_mask = None , tgt_mask = None):
        
        out = self.self_attn(x , x , x , tgt_mask)
        x = self.AddAndNorm1(x + out)

        out = self.self_attn(x , encoder_output , encoder_output , src_mask)
        x = self.AddAndNorm2(x + out)

        out = self.FFN(x)
        x = self.AddAndNorm3(x+out)

        return x

class Decoder(nn.Module):
    def __init__(self , vocab_size , d_model , num_heads , num_neurons , num_encoders):
        super().__init__()

        self.embedd = Embedding(vocab_size , d_model)
        self.positionEmbedd = PossitionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model , num_heads , num_neurons) for _ in range(num_encoders)
            ]
        )
    
    def forward(self , x , Encoder_input , tgt_mask = None , src_mask = None):
        
        # embedding the content in the corpus
        x = self.embedd(x)

        # adding the positional encoding to the embeddings
        x = self.positionEmbedd(x)

        for layer in self.layers:
            x = layer(x , Encoder_input , tgt_mask , src_mask)

        return x   
                 
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
    
    print()
    print("embedding without the positional encoding: ")
    print(Embeddings.shape)

    '''    
    for i, word in enumerate(sentence):
        print(f"{word} = {Embeddings[0, i]}")
    '''
    
    #Embedding with positional encoding

    positional_encod = PossitionalEncoding(d_model , max_len=100)
    PosEncoded = positional_encod(Embeddings)
    
    print()
    print("embedding with the positional encoding: ")
    
    
    '''for i, word in enumerate(sentence):
        print(f"{word} = {PosEncoded[0, i]}")
    '''
    print(PosEncoded.shape)
    print()

    mha = MultiHeadAttention(d_model , 2)
    mha_score = mha(PosEncoded , PosEncoded , PosEncoded)

    print("Multi Head Attention Output")
    print(mha_score.shape)

    encoder = Encoder(vocab_size , d_model , 2 , 10 , 10)
    encoderOut = encoder(token_id)

    print(encoderOut.shape)
