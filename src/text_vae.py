import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from noise import noisy

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.ones_like(std)#torch.randn_like(std)
    return eps.mul(std).add_(mu)

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

class TextModel(nn.Module):
    """Container module with word embedding and projection layers"""
    def __init__(self,  initrange=0.1):
        super().__init__()
        self.embed = nn.Embedding(38926, 512)
        self.proj = nn.Linear(1024, 38926)
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)

class DAE(TextModel):
    """Denoising Auto-Encoder"""
    def __init__(self, latent_dim, vocab):
        super().__init__()
        self.drop = nn.Dropout(0.5)
        self.E = nn.LSTM(512, 1024, 1,
                         dropout=0.2, bidirectional=True)
        self.G = nn.LSTM(512, 1024, 1,
                         dropout=0.2)
        self.h2mu = nn.Linear(2048, latent_dim)
        self.h2logvar = nn.Linear(2048, latent_dim)
        self.z2emb = nn.Linear(latent_dim, 512)
        self.vocab = vocab
        #self.opt = optim.Adam(self.parameters(), lr=0.0005, betas=(0.5, 0.999))

    def flatten(self):
        self.E.flatten_parameters()
        self.G.flatten_parameters()

    def encode(self, input):
        input = self.drop(self.embed(input))
        _, (h, _) = self.E(input)
        hes = torch.cat([h[-2], h[-1]], 1) # torch.squeeze(hes,0)
        return self.h2mu(hes), self.h2logvar(hes)

    def decode(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def forward(self, input, graph_z=None, is_train=False):
        if False:# self.training:
            _input = noisy(self.vocab, input, 0.2) 
        else:
            _input = input
        mu, logvar = self.encode(_input)
        # z = reparameterize(mu, logvar)
        # logits, _ = self.decode(z, input)
        # if src_dict is not None:
        #     self.generate(z, 20, 'greedy', src_dict)
        return mu, logvar, #z, None #logits

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.flatten(),
            ignore_index=1, reduction='none').view(targets.size())
        return loss.mean(dim=0)#loss.sum(dim=0)

    def accuracy(self, logits, targets):
        T, B = targets.size()
        sft = nn.Softmax(dim=2)
        pred_tokens = torch.argmax(sft(logits),2) # T,B
        acc = ((pred_tokens == targets) * (targets != 1)).sum() / (targets != 1).sum()
        # for i in range(len(pred_tokens.t())):
        #     tokens = []; gold_tokens = []
        #     for tok in targets.t()[i]:
        #         tok = tok.item()
        #         if tok < len(src_dict):
        #             gold_tokens.append(src_dict.itos[tok].replace("~", "_"))          
        #     print("\ngold_tokens:", gold_tokens)
        #     for tok in pred_tokens.t()[i]:
        #         tok = tok.item()
        #         if tok < len(src_dict):
        #             tokens.append(src_dict.itos[tok].replace("~", "_"))          
        #     print("tokens:", tokens)
        return acc.item()
    
    def loss(self, losses):
        return losses['rec']

    def autoenc(self, inputs, targets, graph_z=None, is_train=False):
        mu, logvar = self(inputs, graph_z, is_train) # z, logits
        return mu, logvar #z #self.loss_rec(logits, targets).mean(), z

    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.opt.step()

   
    def generate(self, z, max_len, alg):
        assert alg in ['greedy' , 'sample' , 'top5']
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(2)
        hidden = None
        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg == 'sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
            elif alg == 'top5':
                not_top5_indices=logits.topk(logits.shape[-1]-5,dim=2,largest=False).indices
                logits_exp=logits.exp()
                logits_exp[:,:,not_top5_indices]=0.
                input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t()

        predicted_tokens = torch.cat(sents).t()

        # See predictions...
        for i in range(predicted_tokens.size(0)):
            tokens = []; gold_tokens = []

            # for tok in nodetoken_ids[i]:
            #     tok = tok.item()
            #     if tok < len(src_dict):
            #         gold_tokens.append(src_dict.itos[tok].replace("~", "_"))          
            # print("\ngold_tokens:", gold_tokens)
            for tok in predicted_tokens[i]:
                tok = tok.item()
                if tok < len(self.vocab):
                    tokens.append(self.vocab.itos[tok].replace("~", "_"))          
            print("tokens:", tokens)

        return torch.cat(sents)
