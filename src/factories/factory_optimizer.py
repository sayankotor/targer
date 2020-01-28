"""creates various optimizers"""
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class OptimizerFactory():
    """OptimizerFactory contains wrappers to create various optimizers."""
    @staticmethod
    def create(args, tagger, special_bert = False):        
        if args.opt == 'sgd':
            optimizer = optim.SGD(list(tagger.parameters()), lr=args.lr, momentum=args.momentum)
        elif args.opt == 'adam':
            optimizer = optim.Adam(list(tagger.parameters()), lr=args.lr, betas=(0.9, 0.999))
        else:
            raise ValueError('Unknown optimizer, must be one of "sgd"/"adam".')
            
        if (special_bert):
            bert_parameters = list(tagger.word_seq_indexer.emb.parameters())
            not_bert_parameters = list(list(tagger.birnn_layer.parameters()) + list(tagger.lin_layer.parameters()) + list(tagger.log_softmax_layer.parameters()))
            optimizer = optim.Adam([{'params':not_bert_parameters, 'lr':args.lr, 'betas':(0.9, 0.999)}, {'params':bert_parameters, 'lr':args.lr_bert, 'betas':(0.9, 0.999)}])

        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + args.lr_decay * epoch))
        #bert_scheduler = LambdaLR(bert_optimizer, lr_lambda=lambda epoch: 1 / (1 + args.lr_decay * epoch))
        return optimizer, scheduler
