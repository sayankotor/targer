"""class implements context word embeddings, like Elmo, Bert"""
"""The meaning of the equal word can change in different context, in different batch"""
import torch.nn as nn
from src.layers.layer_base import LayerBase
from allennlp.modules.elmo import Elmo, batch_to_ids



class LayerContextWordEmbeddingsBert(LayerBase):
    """LayerWordEmbeddings implements word embeddings."""
    def __init__(self, word_seq_indexer, gpu, freeze_word_embeddings=False, tpnm = "Elmo", pad_idx=0):
        super(LayerContextWordEmbeddingsBert, self).__init__(gpu)
        print ("LayerContextWordEmbeddings dert init")
        self.embeddings = word_seq_indexer.emb
        self.embeddings.padding_idx = pad_idx
        self.word_seq_indexer = word_seq_indexer
        self.embeddings_dim = 768#self.embeddings.get_output_dim()
        self.output_dim = self.embeddings_dim
        self.gpu = gpu
        self.tpnm = "Elmo"

    def is_cuda(self):
        return self.embeddings.weight.is_cuda
    
    def to_gpu(self, tensor):
        if self.gpu > -1:
            return tensor.cuda(device=1)
        else:
            return tensor.cpu()

    def forward(self, word_sequences):
        print ("forward")
        tokens_tensor, segments_tensor = self.word_seq_indexer.batch_to_ids(word_sequences)
        tokens_tensor = self.to_gpu(tokens_tensor)
        segments_tensor = self.to_gpu(segments_tensor)
        print (tokens_tensor.shape)
        print (segments_tensor.shape)
        
        encoded_layers, _ = self.embeddings(tokens_tensor, segments_tensor)
        
        token_embeddings = []
        for token_i in range(len(tokenized_text)):  
            hidden_layers = [] 
            for layer_i in range(len(encoded_layers)):
                vec = encoded_layers[layer_i][0][token_i]
                hidden_layers.append(vec)
                token_embeddings.append(hidden_layers)

        summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]
        
        print ("len(summed_last_4_layers)", len(summed_last_4_layers))
        print ("len emb", len(summed_last_4_layers[1]))     
        exit()   
        return summed_last_4_layers
