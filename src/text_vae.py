
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from noise import noisy

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

class TextModel(nn.Module):
    """Container module with word embedding and projection layers"""
    def __init__(self, vocabsize, initrange=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocabsize, 512)
        self.proj = nn.Linear(1024, vocabsize)
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)

class DAE(TextModel):
    """Denoising Auto-Encoder"""
    def __init__(self, latent_dim, vocabsize, padidx):
        super().__init__(vocabsize)
        self.drop = nn.Dropout(0.5)
        self.E = nn.LSTM(512, 1024, 1, #int(latent_dim/2)
                         dropout=0.0, bidirectional=True)
        self.G = nn.LSTM(512, 1024, 1,
                          dropout=0.0)
        
        self.hgen = nn.Linear(2048, latent_dim)
        # self.h2mu = nn.Linear(2048, latent_dim)
        # self.h2logvar = nn.Linear(2048, latent_dim)
        self.z2emb = nn.Linear(latent_dim, 512)

        self.padidx = padidx


    def flatten(self):
        self.E.flatten_parameters()
        self.G.flatten_parameters()

    def encode(self, input):
        input = self.drop(self.embed(input))
        _, (ht, _) = self.E(input)
        #breakpoint()
        hes = torch.cat([ht[-2], ht[-1]], 1) # torch.squeeze(hes,0)
        return self.hgen(hes) #self.h2mu(hes), self.h2logvar(hes)

    def decode(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def forward(self, input):
        if self.training:
            _input = noisy(self.padidx, input, 0.3, 0, 0, 0) 
        else:
            _input = input
        #mu, logvar = self.encode(_input)
        h = self.encode(_input)
        return h #mu, logvar

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.flatten(), ignore_index=self.padidx, reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def accuracy(self, logits, targets, is_dev=False):
        T, B = targets.size()
        sft = nn.Softmax(dim=2)
        pred_tokens = torch.argmax(sft(logits),2) # T,B
        #pred_tokens = torch.argmax(logits,-1) # T,B
        
        # if is_dev:
        #     target_tokens = [];
        #     for tok in targets.t()[0]:
        #         tok = tok.item()
        #         target_tokens.append(self.vocab.itos[tok])#.replace("~", "_"))          
        #     print("target_tokens:", target_tokens)
        #     tokens = []; 
        #     for tok in pred_tokens.t()[0]:
        #         tok = tok.item()
        #         tokens.append(self.vocab.itos[tok])#.replace("~", "_"))          
        #     print("pred_tokens:", tokens)
        #     print('=====')

        acc = ((pred_tokens == targets) * (targets != self.padidx)).sum() / (targets != self.padidx).sum()
        return acc.item()

    def autoenc(self, inputs, targets):
        #mu, logvar = self(inputs) 
        return self(inputs)
   
    def generate(self, z, max_len, alg):
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(2)#self.vocab.go)
        hidden = None
        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg == 'sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()

        #breakpoint()
        # See predictions...
        #predicted_tokens = torch.cat(sents).t()
        # for i in range(predicted_tokens.size(0)):
        #tokens = []; 
        #for tok in predicted_tokens[0]:
            #tok = tok.item()
            #if tok < len(self.vocab):
                #tokens.append(self.vocab.itos[tok])#.replace("~", "_"))          
            #tokens.append(self.vocab.itos[tok])#.replace("~", "_"))          
            #tokens.append(self.vocab.idx2word[tok])#.replace("~", "_"))          
        #print("tokens:", tokens)
        #return torch.cat(sents)
        return sents


        
class AAE(DAE):
    """Adversarial Auto-Encoder"""

    def __init__(self, latent_dim, vocab):
        super().__init__(latent_dim, vocab)
        self.D = nn.Sequential(nn.Linear(32, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid())
        #self.optD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def loss_adv(self, z):
        zn = torch.randn_like(z)
        zeros = torch.zeros(len(z), 1, device=z.device)
        ones = torch.ones(len(z), 1, device=z.device)
        loss_d = F.binary_cross_entropy(self.D(z.detach()), zeros) + \
            F.binary_cross_entropy(self.D(zn), ones)
        loss_g = F.binary_cross_entropy(self.D(z), ones)
        return loss_d, loss_g

    
    
