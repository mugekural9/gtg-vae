import transformer as nmt_model
import torch, time
import utils.constants as Constants
from dataset import make_text_iterator_from_file, build_dataset, OrderedIterator, make_features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_translator(opt):
  fields, model = nmt_model.load_test_model(opt)                                                                 
  model.to(device)
  translator = Translator(model, fields, opt)
  return translator


class Translator(object):
  def __init__(self, model, fields, opt, out_file=None):
    self.model = model
    self.fields = fields
    # self.gpu = opt["gpu"]
    # self.cuda = opt["gpu"] > -1
    # self.device = torch.device('cuda' if self.cuda else 'cpu')
    # self.out_file = out_file
    # self.tgt_eos_id = fields["tgt"].vocab.stoi[Constants.EOS_WORD]
    # self.tgt_bos_id = fields["tgt"].vocab.stoi[Constants.BOS_WORD]
    # self.tgt2_eos_id = fields["tgt2"].vocab.stoi[Constants.EOS_WORD]
    # self.tgt2_bos_id = fields["tgt2"].vocab.stoi[Constants.BOS_WORD2]
    # self.src_eos_id = fields["src"].vocab.stoi[Constants.EOS_WORD]
    

  # def build_tokens(self, idx, side="tgt"):
  #   assert side in ["src", "tgt", "tgt2"], "side should be either src or tgt"
  #   vocab = self.fields[side].vocab
  #   if side == "tgt":
  #     eos_id = self.tgt_eos_id
  #   if side == "tgt2":
  #     eos_id = self.tgt2_eos_id
  #   else:
  #     eos_id = self.src_eos_id
  #   tokens = []
  #   for tok in idx:
  #     if tok == eos_id:
  #       break
  #     if tok < len(vocab):
  #       tokens.append(vocab.itos[tok].replace("~", "_"))
  #   return tokens

  # def translate(self, src_data_iter, tgt_data_iter, batch_size, out_file=None):
  #   data = build_dataset(self.fields,
  #                        src_data_iter=src_data_iter,
  #                        tgt_data_iter=tgt_data_iter,
  #                        use_filter_pred=False)

  #   def sort_translation(indices, translation):
  #     ordered_transalation = [None] * len(translation)
  #     for i, index in enumerate(indices):
  #       ordered_transalation[index] = translation[i]
  #     return ordered_transalation

  #   if self.cuda:
  #       cur_device = "cuda"
  #   else:
  #       cur_device = "cpu"

  #   data_iter = OrderedIterator(
  #     dataset=data, device=cur_device,
  #     batch_size=batch_size, train=False, sort=True,
  #     sort_within_batch=True, shuffle=True)

  #   start_time = time.time()
  #   batch_count = 0
  #   all_translation = []

  #   for batch in data_iter:
  #     word_preds = self.translate_batch(batch)[:,0].tolist()
  #     tran_words = self.build_tokens(word_preds, side='tgt2')
  #     tran = ' '.join(tran_words)
      
      

  # def translate_batch(self, batch):
  #   src_seq = make_features(batch, 'src')
  #   src_len = src_seq.size(0)
  #   src_enc = torch.rand(1, 1, 512).to("cuda") #  (ALWAYS1, B, Zdim)

  #   self.model.decoder.init_state(src_seq, src_enc)
  #   dec_output, *_ = self.model.decoder(src_seq, step=None) # src_seq as tgt_seq, but any need for shifting?
  #   word_prob = self.model.generator(dec_output.squeeze(0))
  #   top_probs =  torch.argmax(word_prob, 2)  # (Ty, B)
  #   return top_probs
  

