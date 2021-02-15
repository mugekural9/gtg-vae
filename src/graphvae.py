import torch
from torch_geometric.nn import GCNConv, InnerProductDecoder
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import global_max_pool


MAX_LOGSTD = 10
EPS = 1e-15

class GraphVAE(torch.nn.Module):
    """
    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    
    def __init__(self, input_dim, hidden_dim, nmax, nodeclass_num):
       super(GraphVAE, self).__init__()
       self.encoder = VGAE_encoder(input_dim, out_channels=hidden_dim)
       self.inner_product_decoder = InnerProductDecoder() 
       self.decoder = VGAE_decoder(hidden_dim, nmax,  nodeclass_num)
   
    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, batch, *args, **kwargs):
        """"""
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)

        # self.__mu__ = global_max_pool(self.__mu__, batch) # (B,Zdim)
        # self.__logstd__ = global_max_pool(self.__logstd__, batch) # (B,Zdim)
       
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    
    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


    def recon_loss(self, f_hat, feat_hat, x, pos_edge_index, batch, nodetoken_ids, src_dict, node_dict):
        # pos_edge_index: (2, num_of_edges)
        # f_hat: (B, maxnode, nodelabelsize)
        # feat_hat: (B, maxnode, feature_dim)
        # x: (maxnode, nodefeatures)
        # nodetoken_ids: (B, maxnode)

        nodelabel_loss= 0
        pos_loss = 0
        neg_loss = 0
        kl_loss =  0
        recon_loss = 0
        
        batch_size = f_hat.size(0)
        maxnode = f_hat.size(1)

        # F_hat (Nodelabel prediction) loss
        # Create indexing matrix for batch: [batch, 1]

        # batch_index = torch.arange(0, batch_size).view(batch_size, 1)
        # node_index  = torch.arange(0, maxnode).view(1, maxnode).repeat(batch_size,1)
        # m =  nn.Softmax(dim=2)
        # f_hat = m(f_hat)
        # nodelabel_probs = f_hat[batch_index, node_index, nodetoken_ids.data]
        # nodelabel_loss = -torch.log(nodelabel_probs + 1e-15).mean()

        # See predictions...
        # predicted_node_tokens = torch.argmax(f_hat,2) # (B, maxnode)
        # correct_predicted_node_tokens = (predicted_node_tokens == nodetoken_ids).sum()
        # for i in range(len(predicted_node_tokens)):
        #     tokens = []; gold_tokens = []
        #     #i=0
        #     #print("i:..",i)
        #     for tok in predicted_node_tokens[i]:
        #         tok = tok.item()
        #         if tok < len(src_dict):
        #             tokens.append(src_dict.itos[tok].replace("~", "_"))          
        #     print("tokens:", tokens)
        #     for tok in nodetoken_ids[i]:
        #         tok = tok.item()
        #         if tok < len(src_dict):
        #             gold_tokens.append(src_dict.itos[tok].replace("~", "_"))          
        #     print("gold_tokens:", gold_tokens)
       
        # A_hat loss 
        feat_hat = feat_hat.view(x.size(0), -1)
        pos_loss = -torch.log(
            self.inner_product_decoder(feat_hat, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        neg_edge_index = negative_sampling(pos_edge_index, feat_hat.size(0))
        neg_loss = -torch.log(1 -
                              self.inner_product_decoder(feat_hat, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        roc_auc_score, avg_precision_score = self.test(feat_hat, pos_edge_index, neg_edge_index)
        kl_loss = 1 / x.size(0) * self.kl_loss()
        correct_predicted_node_tokens = None

        return kl_loss, pos_loss, neg_loss, roc_auc_score, avg_precision_score   # nodelabel_loss, correct_predicted_node_tokens
 

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.inner_product_decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.inner_product_decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAE_encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
       super(VGAE_encoder, self).__init__()
       self.hidden_channels = 2*out_channels
       self.conv1 = GCNConv(in_channels, self.hidden_channels)
       self.conv2_mu  = GCNConv( self.hidden_channels, out_channels)
       self.conv2_sig = GCNConv( self.hidden_channels, out_channels)


    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x_mu = self.conv2_mu(x,edge_index)
        x_sig = self.conv2_sig(x,edge_index)

        return x_mu,x_sig



# GraphVAE (ref: Simonovsky et al)
class VGAE_decoder(torch.nn.Module):
    def __init__(self, in_channels, nmax, nodeclass_num):
       super(VGAE_decoder, self).__init__()
       self.in_channels = in_channels
       self.nmax = nmax
       
       self.nodeclass_num = nodeclass_num
       self.nodeclass_linear = Linear(in_channels, nodeclass_num*nmax)
       self.nodefeatures_linear = Linear(in_channels, in_channels*nmax)
            

    def forward(self, z):
        
        F_hat = self.nodeclass_linear(z)
        F_hat = torch.reshape(F_hat, (-1, self.nmax, self.nodeclass_num))     # GinBatch, nmax, nodeclass_num

        Feat_hat = self.nodefeatures_linear(z)
        Feat_hat = torch.reshape(Feat_hat, (-1, self.nmax, self.in_channels)) # Ginbatch, nmax, feature_dim
        return F_hat, Feat_hat 

