import torch
import numpy as np
import pandas as pd
import logging
import torch.nn as nn
import math

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, BertTokenizerFast, BertModel, AdamW, TFBertModel
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.modeling_bert import BertEmbeddings, BertSelfAttention
from torch.utils.data import Dataset, DataLoader
from keras.preprocessing.sequence import pad_sequences

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

COL_NAMES = ['TopNumber', 'AirlineName','ReviewerName','Rating','ReviewDate','ReviewTitle',\
             'ReviewText','Tags', 'DateofTravel', 'Aspects', 'ResponserName', 'ResponseDate', 'ResponseText', 'ReviewerProfileUrl',\
             'AirlineNation', 'CrawlTime']

PRE_TRAINED = 'bert-base-uncased'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ASPECT_NAMES = ['LEG', 'SIT', 'ENT', 'CUS', 'VOM', 'CLE', 'CKI', 'FNB']
VOCAB_DIC = BertTokenizerFast.from_pretrained(PRE_TRAINED).get_vocab()
TOPN = 150


# This one is implemented with weight loss per class            
class BertBonzWeightLoss(BertModel):
    def __init__(self, config):
        super(BertBonzWeightLoss, self).__init__(config)
        self.config = config
        self.embeddings.llr_embeddings = nn.ModuleList(nn.Embedding(4, 768, 3) for _ in range(len(ASPECT_NAMES)))
        self.classifier = nn.Linear(768, config.num_aspect*3)
        self.init_weights()
        self.embeddings.llr_embeddings.apply(self._xavier)
        self.pooler.apply(self._xavier)
        self.classifier.apply(self._xavier)
        
    def forward(self, 
                input_ids=None, 
                llr_ids=None, 
                labels=None, 
                token_type_ids=None, 
                position_ids=None,
                weight_loss=None):
        # BERT EMBEDDINGS NEW
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        device = input_ids.device
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        inputs_embeds = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        
        if llr_ids is not None:
            temp = [self.embeddings.llr_embeddings[i](llr_ids[:,i,:]) for i in range(self.config.num_aspect)]
            llr_embeddings = sum(temp)
        else:
            llr_embeddings = torch.zeros(inputs_embeds.size(), device=device).fill_(3).long()
        
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + llr_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        
        
        # BERT ENCODER
        encoder_outputs = self.encoder(
            embeddings,
            attention_mask=None,
            head_mask=[None]*12,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=self.config.output_attentions
        )
        sequence_output = encoder_outputs[0]
        
        # CLASSIFIER
        CLS_token = sequence_output[:,0]
        predict = self.classifier(CLS_token)
        
        loss_fn = nn.functional.cross_entropy
        if labels is not None:
            if weight_loss is None:
                loss = loss_fn(predict.view(input_shape[0], 3,-1), labels)
            else:
                loss = torch.tensor(0).float().to(DEVICE)
                for asp_i in range(len(ASPECT_NAMES)):
                    loss += loss_fn(predict.view(input_shape[0], 3,-1)[:,:,asp_i], labels[:,asp_i], weight_loss[asp_i, :])
                loss /= len(ASPECT_NAMES)
                    
            outputs = (predict.view(input_shape[0], 3,-1), loss, CLS_token, sequence_output) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        else:
            outputs = (predict.view(input_shape[0], 3,-1), CLS_token, sequence_output) + encoder_outputs[1:]
        return outputs
    
    
    def load_pretrained_weight(self):
        sd = self.state_dict()
        sd_bert_pretrained = BertModel.from_pretrained(PRE_TRAINED).state_dict()
        for k in sd_bert_pretrained.keys():
            if k in sd.keys():
                sd[k] = sd_bert_pretrained[k]
        self.load_state_dict(sd)
        print('Succesfully load pre-trained weights')
        
    def llr_embed_pad(self):
        for i in range(len(self.embeddings.llr_embeddings)):
            temp = self.embeddings.llr_embeddings[i].weight.data
            temp[-1,:] = torch.zeros(temp.size(1))
        
    def _xavier(self, module):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                param.data.zero_()
                
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
                
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.embeddings.llr_embeddings.parameters():
            param.requires_grad = True
        for param in self.pooler.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True            
    

class BonzDataset(Dataset):
    def __init__(self, data, llr_words):
        self.input_ids = torch.LongTensor(list(data.input_ids))
        self.llr_embeddings = torch.LongTensor(list(data.llr_embeddings))
        if 'llr_embeddings' in data.columns:
            self.llr_embeddings = torch.LongTensor(list(data.llr_embeddings))
        else:
            self.llr_embeddings = torch.zeros(data.shape[0],1).long()
        if 'labels' in data.columns:
            self.labels = torch.LongTensor(list(data.labels))
        else:
            self.labels = None
        self.llr_words = llr_words
        
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        if self.labels is None:
            outputs = (self.input_ids[idx], self.llr_embeddings[idx])
        else:
            outputs = (self.input_ids[idx], self.llr_embeddings[idx], self.labels[idx])
        
        return outputs
    

    


