import torch
import torch.nn as nn
import re

from torch.nn.init import xavier_uniform
from transformer_decoder import TransformerDecoder
from transformer_encoder import TransformerEncoder
from embeddings import Embeddings
from dataset import load_fields_from_vocab, save_fields_to_vocab
 
from utils.misc import use_gpu
import utils.constants as Constants
import pdb

from torch.nn import Linear
from torch.nn.init import xavier_uniform_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class JointModel(nn.Module):
  def __init__(self, encoder, hidden_dim, out_dim):
    super(JointModel, self).__init__()                                                                                               
    self.encoder   = encoder                                                                                                           
    self.mu_linear = Linear(hidden_dim, out_dim)
    self.std_linear= Linear(hidden_dim, out_dim)
    # self.decoder = decoder
    
  def forward(self, src, tgt=None, lengths=None):                                                                                   
    # tgt = tgt[:-1]  # exclude last target from inputs
    _, memory_bank, lengths = self.encoder(src, lengths)
    
    # self.decoder.init_state(src, memory_bank)                                                                                        
    # dec_out, attns = self.decoder(tgt)                                                                                               
    memory_mu = self.mu_linear(memory_bank)
    memory_std = self.std_linear(memory_bank)
    z = self.reparametrize(memory_mu, memory_std)
    return z #memory_bank # dec_out, attns    


  def reparametrize(self, mu, logstd):
    if self.training:
      return mu + torch.randn_like(logstd) * torch.exp(logstd)
    else:
      return mu

  
def load_test_model(opt): 
  model_path = opt["model_path"]
  print("loading pretrained transformer from...", model_path)
  checkpoint = torch.load(model_path,
                          map_location=lambda storage,
                          loc:storage)

  fields = load_fields_from_vocab(checkpoint['vocab'])                                                                               
  model = build_base_model(opt, fields, use_gpu(opt), checkpoint)

  #model_opt = checkpoint['opt']
  #model.eval()                                                                                                                       
  #model.generator.eval()                                                                                                             
  return fields, model


def build_embeddings(opt, word_dict, for_encoder=True):                                                                              
  """                                                                                                                                
  Build an Embeddings instance.                                                                                                      
  Args:                                                                                                                              
      opt: the option in current environment.                                                                                        
      word_dict(Vocab): words dictionary.                                                                                            
      feature_dicts([Vocab], optional): a list of feature dictionary.                                                                
      for_encoder(bool): build Embeddings for encoder or decoder?                                                                    
  """                                                                                                                                
  if for_encoder:                                                                                                                    
    embedding_dim = opt["word_emb_size"]                                                                                            
  else:                                                                                                                              
    embedding_dim = opt["word_emb_size"]                                                                                            
  word_padding_idx = word_dict.stoi[Constants.PAD_WORD]                                                                              
  num_word_embeddings = len(word_dict)                                                                                               
  return Embeddings(word_vec_size=embedding_dim,                                                                                     
                    position_encoding=opt["position_encoding"],                                                                         
                    dropout=opt["dropout"],                                                                                             
                    word_padding_idx=word_padding_idx,                                                                               
                    word_vocab_size=num_word_embeddings,                                                                             
                    sparse=False) #opt["optim"] == "sparseadam")     

def build_encoder(opt, embeddings):
  """
  Various encoder dispatcher function.
  Args:
      opt: the option in current environment.
      embeddings (Embeddings): vocab embeddings for this encoder.
  """
  return TransformerEncoder(opt["enc_layers"], opt["d_model"],
                            opt["heads"], opt["transformer_ff"],
                            opt["dropout"], embeddings)

def build_decoder(opt, embeddings):                                                                                                  
  """                                                                                                                                
  Various decoder dispatcher function.                                                                                               
  Args:                                                                                                                              
      opt: the option in current environment.                                                                                        
      embeddings (Embeddings): vocab embeddings for this decoder.                                                                    
  """                                                                                                                                
  return TransformerDecoder(opt["dec_layers"], opt["d_model"],                                                                        
                     opt["heads"], opt["transformer_ff"],                                                                                  
                     opt["dropout"], embeddings)    



def build_base_model(model_opt, fields, gpu, checkpoint=None):                                                                       
  """                                                                                                                                
  Args:                                                                                                                              
      model_opt: the option loaded from checkpoint.                                                                                  
      fields: `Field` objects for the model.                                                                                         
      gpu(bool): whether to use gpu.                                                                                                 
      checkpoint: the model gnerated by train phase, or a resumed snapshot                                                           
                  model from a stopped training.                                                                                     
  Returns:                                                                                                                           
      the NMTModel.                                                                                                                  
  """                                                                                                                                

  # Build encoder.                                                                                                                   
  src_dict = fields["src"].vocab
  src_embeddings = build_embeddings(model_opt, src_dict)                                                                             
  encoder = build_encoder(model_opt, src_embeddings)
  
  
  # Build decoder.                                                                                                                   
  # tgt_dict = fields["tgt"].vocab                                                                                                     
  # tgt_embeddings = build_embeddings(model_opt, tgt_dict,                                                                             
  #                                   for_encoder=False)                                                                               

  # # Share the embedding matrix - preprocess with share_vocab required.                                                               
  # if model_opt["share_embeddings"]:                                                                                                     
  #   # src/tgt vocab should be the same if `-share_vocab` is specified.                                                               
  #   if src_dict != tgt_dict:                                                                                                         
  #     raise AssertionError('The `-share_vocab` should be set during '                                                                
  #                          'preprocess if you use share_embeddings!')                                                                
  #   tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight                                                                  

  # decoder = build_decoder(model_opt, tgt_embeddings)                                                                                 
  # Build partial NMTModel(only decoder).                                                                                             
  model = JointModel(encoder, model_opt["d_model"], model_opt["d_graphz"]) # decoder                                                                                               

  # Build Generator.                                                                                                                 
  # gen_func = nn.LogSoftmax(dim=-1)                                                                                                   
  # generator = nn.Sequential(                                                                                                         
  #  nn.Linear(model_opt["d_model"], len(fields["tgt"].vocab), bias=False),                                                         
  #  gen_func                                                                                                                         
  # )                                                                                                                                  
  # if model_opt["share_decoder_embeddings"]:                                                                                             
  #  generator[0].weight = decoder.embeddings.word_lut.weight
  
  # Load the model states from checkpoint or initialize them.                                                                        
  if checkpoint is not None:                                                                                                         
    print("WEIGHTS ARE FROM S2SPARSER")
    # This preserves backward-compat for models using customed layernorm                                                             
    def fix_key(s):                                                                                                                  
      s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',                                                                                  
                 r'\1.layer_norm\2.bias', s)                                                                                         
      s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',                                                                                  
                 r'\1.layer_norm\2.weight', s)                                                                                       
      return s                                                                                                                       

    checkpoint['model'] = {fix_key(k): v for (k, v) in checkpoint['model'].items()}                                                                      
    # end of patch for backward compatibility                                                                                        
    model.load_state_dict(checkpoint['model'], strict=False)                                                                         
    # generator.load_state_dict(checkpoint['generator'], strict=False)
  else:
    #breakpoint()
    if False: #model_opt["param_init"] != 0.0:
      for p in model.parameters():                                                                                                   
        p.data.uniform_(-model_opt["param_init"], model_opt["param_init"])                                                                 
      for p in generator.parameters():                                                                                               
        p.data.uniform_(-model_opt["param_init"], model_opt["param_init"])                                                                 
    if model_opt["param_init_glorot"]:                                                                                                  
      for p in model.parameters():                                                                                                   
        if p.dim() > 1:                                                                                                              
          xavier_uniform_(p)                                                                                                         
      # for p in generator.parameters():                                                                                               
      #   if p.dim() > 1:                                                                                                              
      #     xavier_uniform_(p)                                                                                                         
    # if hasattr(model.encoder, 'embeddings'):                                                                                         
    #   model.encoder.embeddings.load_pretrained_vectors(                                                                              
    #       model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)                                                                  

    # if hasattr(model.decoder, 'embeddings'):                                                                                         
    #   model.decoder.embeddings.load_pretrained_vectors(                                                                              
    #       model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)                                                                  
      #pdb.set_trace()                                                                                                                   

  # Add generator to model (this registers it as parameter of model).                                                                
  # model.generator = generator                                                                                                        
  model.to(device)                                                                                                                  
  return model          


