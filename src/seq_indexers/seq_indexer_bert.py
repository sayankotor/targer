"""indexer for BERT model"""
"""join list of input words into string"""
"""provide BERT tokenization"""

import string
import re
from src.seq_indexers.seq_indexer_base_embeddings import SeqIndexerBaseEmbeddings
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import torch


class SeqIndexerBert(SeqIndexerBaseEmbeddings):
    """SeqIndexerWord converts list of lists of words as strings to list of lists of integer indices and back."""
    def __init__(self, gpu=-1, check_for_lowercase=True, embeddings_dim=0, verbose=True, path_to_pretrained = "/home/vika/targer/pretrained", bert_type = 'bert-base-uncased'):
        SeqIndexerBaseEmbeddings.__init__(self, gpu=gpu, check_for_lowercase=check_for_lowercase, zero_digits=True,
                                          pad='<pad>', unk='<unk>', load_embeddings=True, embeddings_dim=embeddings_dim,
                                          verbose=verbose, isBert = True)
        print ("create seq indexer BERT!")
        #self.no_context_base = False
        self.bert = True
        self.path_to_pretrained = path_to_pretrained
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.emb = BertModel.from_pretrained(path_to_pretrained)
        print ("Bert model loaded succesifully")
        
    def batch_to_ids(self, batch):
        print ("batch to ids bert")
        print (batch[:3])
        tokenized_texts = [self.tokenizer.tokenize(''.join(sent)) for sent in batch]
        MAX_LEN = np.max(np.array([len(seq) for seq in tokenized_texts]))
        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        tokens_tensor = torch.tensor(input_ids)
        segments_tensor = torch.tensor(np.ones(input_ids.shape)).to(torch.int64)
        print (tokens_tensor[:3])
        print (segments_tensor[:3])
        print ("end_to_batch to ids bert")
        return tokens_tensor, segments_tensor
        