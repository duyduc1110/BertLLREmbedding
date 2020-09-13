# https://medium.com/udacity-pytorch-challengers/ideas-on-how-to-fine-tune-a-pre-trained-model-in-pytorch-184c47185a20

import re
import torch
import numpy as np
import pandas as pd
import logging
import time
import torch.nn as nn
import math
import ast

from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, BertTokenizerFast, BertModel, AdamW, TFBertModel
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.modeling_bert import BertEmbeddings, BertSelfAttention
from torch.utils.data import Dataset, DataLoader
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from apex import amp, optimizers
from tqdm.auto import tqdm, trange

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


PRE_TRAINED = 'bert-base-uncased'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ASPECT_NAMES = ['LEG', 'SIT', 'ENT', 'CUS', 'VOM', 'CLE', 'CKI', 'FNB']
VOCAB_DIC = BertTokenizerFast.from_pretrained(PRE_TRAINED).get_vocab()
TOPN = 50


class BertBonz(BertModel):
    def __init__(self, config):
        super(BertBonz, self).__init__(config)
        self.config = config
        self.embeddings.llr_embeddings = nn.ModuleList(nn.Embedding(4, 768, 3) for _ in range(len(ASPECT_NAMES)))
        self.classifier = nn.Linear(768, config.num_aspect*3)
        self.init_weights()
        
        
    def forward(self, 
                input_ids=None, 
                llr_ids=None, 
                labels=None, 
                token_type_ids=None, 
                position_ids=None):
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
        
        loss_fn = nn.CrossEntropyLoss()
        if labels is not None:
            loss = loss_fn(predict.view(input_shape[0], 3,-1), labels)
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
        for i in range(len(ASPECT_NAMES)):
            temp = self.embeddings.llr_embeddings[i].weight.data
            temp[-1,:] = torch.zeros(temp.size(1))

            
# This one is implemented with weight loss per class            
class BertBonzWeightLoss(BertModel):
    def __init__(self, config):
        super(BertBonzWeightLoss, self).__init__(config)
        self.config = config
        self.embeddings.llr_embeddings = nn.ModuleList(nn.Embedding(4, 768, 3) for _ in range(len(ASPECT_NAMES)))
        self.classifier = nn.Linear(768, config.num_aspect*3)
        self.init_weights()
        
        
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
        for i in range(len(ASPECT_NAMES)):
            temp = self.embeddings.llr_embeddings[i].weight.data
            temp[-1,:] = torch.zeros(temp.size(1))
        


class BonzDataset(Dataset):
    def __init__(self, data, llr_words):
        self.input_ids = torch.LongTensor(list(data.input_ids))
        self.llr_embeddings = torch.LongTensor(list(data.llr_embeddings))
        if 'labels' in data.columns:
            self.labels = torch.LongTensor(list(data.labels))
        else:
            self.labels = None
        self.llr_words = llr_words
        
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        '''
        tokens = self.data.input_ids[idx]
        
        llr_embedding = []
        for aspect in ASPECT_NAMES:
            temp = [3] * tokens.shape[0]
            for j in range(tokens.shape[0]):
                for class_, wordlist in llr_words[aspect].items():
                    if tokens[j] in wordlist:
                        temp[j] = class_
                        break
            llr_embedding.append(temp)
        
        llr_embedding = torch.stack([torch.LongTensor(i) for i in llr_embedding], 0)
        
        
        outputs = (torch.LongTensor(tokens), llr_embedding)
        
        if 'labels' in self.data.columns:
            outputs = (torch.LongTensor(tokens), llr_embedding, torch.LongTensor(self.data.labels[idx]))
        '''
        if self.labels is None:
            outputs = (self.input_ids[idx], self.llr_embeddings[idx])
        else:
            outputs = (self.input_ids[idx], self.llr_embeddings[idx], self.labels[idx])
        
        return outputs
    

    
def split_aspect(data):
    temp = np.full((8, data.shape[0]), 2, np.int)
    for idx in range(data.shape[0]):
        aspect = data[idx]
        for i, asp in enumerate(['Legroom', 'Seat', 'Entertainment', 'Customer', 'Value', 'Cleanliness', 'Check-in', 'Food']):
            for sub_asp in aspect:
                if asp in sub_asp:
                    pol = int(sub_asp[-1])
                    temp[i, idx] = 1 if pol > 3 else 0
                    break
    return temp
            

def tokenize_data(data):
    tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED)
    input_ids = tokenizer(list(data))['input_ids']
    input_ids = pad_sequences(input_ids, maxlen=512, padding='post', truncating='post')
    
    return list(input_ids)
    
    
def get_data():
    col_names = ['TopNumber', 'AirlineName','ReviewerName','Rating','ReviewDate','ReviewTitle',\
                 'ReviewText','Tags', 'DateofTravel', 'Aspects', 'ResponserName', 'ResponseDate', 'ResponseText', 'ReviewerProfileUrl',\
                 'AirlineUrl','CrawlTime']
    raw_data = pd.read_csv('./data/airline.txt', sep='\t', header=None, names=col_names)
    data = raw_data[['ReviewText', 'Rating', 'Aspects']]
    data = data[data['Aspects'] != 'No filling in'].reset_index(drop=True) # Filter none aspects
    data.Aspects = data.Aspects.str.split('|').values
    
    '''Split aspects to new columns'''
    aspects_splitted = split_aspect(data.Aspects.values)
    for i in range(len(ASPECT_NAMES)):
        data[ASPECT_NAMES[i]] = aspects_splitted[i,:]
        
    data['input_ids'] = tokenize_data(data.ReviewText.values) # Generate input_ids from review text
    
    return data


def word_class_freq(data, aspect_name, aspect_class=3):
    temp = np.zeros((33000, aspect_class), np.int)
    ids = data.input_ids.values
    labels = data[aspect_name].values

    for sub_ids, sub_lb in zip(ids, labels):
        set_ids = set(sub_ids)
        for ids in set_ids:
            temp[ids, sub_lb] += 1
    
    return temp


def calculate_llr(temp_df, labels):
    N = data.shape[0]
    total_scores = []

    for i in temp_df.index.values:
        llr_scores = []
        for class_ in [0,1,2]:
            num_class_doc = np.sum(labels == class_)
            n11 = temp_df.loc[i, class_]
            n10 = num_class_doc - n11
            n01 = temp_df.loc[i, 'total'] - n11
            n00 = (N - n11 - n10 - n01)
            pt = (1e-10 + n11 + n01)/N
            p1 = n11/(1e-10 + n11 + n10)
            p2 = n01/(1e-10 + n01 + n00)


            try:
                e1 = n11 * (math.log(pt) - math.log(p1))
            except:
                e1 = 0
            try:
                e2 = n10 * (math.log(1-pt) - math.log(1-p1))
            except:
                e2 = 0
            try:
                e3 = n01 * (math.log(pt) - math.log(p2))
            except:
                e3 = 0
            try:
                e4 = n00 * (math.log(1-pt) - math.log(1-p2))
            except:
                e4 = 0

            llr_score = -2 * (e1+e2+e3+e4)
            if n11 < n01:
                llr_score = 0
            llr_scores.append(llr_score)

        total_scores.append(llr_scores)
    
    llr_df = pd.DataFrame(np.array(total_scores), index=temp_df.index, columns=temp_df.columns.values[:-1])

    return llr_df


def generate_llr_score(data, aspect):
    temp = word_class_freq(data, aspect)
    
    temp_df = pd.DataFrame(temp)
    temp_df['total'] = np.sum(temp, -1)
    temp_df = temp_df[temp_df['total'] != 0]
    temp_df = temp_df.drop(0,0)
    
    return calculate_llr(temp_df, data[aspect].values)





data = pd.read_csv('sample.csv', sep='\t', index_col=0)

for col in tqdm(['input_ids', 'labels', 'llr_embeddings']):
    data[col] = [ast.literal_eval(i) for i in tqdm(data[col].values)]
    
    
weight_loss = []
for aspect in ASPECT_NAMES:
    temp = 1/data.loc[:, aspect].value_counts(0, 0).values
    weight_loss.append(temp.tolist())
    
weight_loss = torch.tensor(weight_loss).to(DEVICE)


config = BertConfig.from_pretrained('bert-base-uncased')
config.num_aspect = len(ASPECT_NAMES)
model = BertBonzWeightLoss(config)
model.to(DEVICE)

model.load_pretrained_weight() # Load pre-trained BERT weights for BERT's layers 
model.llr_embed_pad() # Set LLR embedding padding idx to 0-value tensor



''' Using apex for faster training
optimizer_list = []
for i in range(10):
    optimizer_list.append(AdamW(model.parameters(), lr=3e-5, correct_bias=False))

model = amp.initialize(model, opt_level="O2", verbosity=0)
''' 

''' Save origin state dict of Model and Optimizer'''
torch.save(model.state_dict(), 'origin_sd.pth')
origin_sd = torch.load('origin_sd.pth')


# Training with K-fold
new_data = data.sample(frac=1).reset_index(drop=True)
kf = KFold(2)
BATCH_SIZE = 7
EPOCH = 5
LEARNING_RATE = 2e-5


last_predict = []
i = 0
for train_idx, test_idx in tqdm(kf.split(new_data)):
    train_data = new_data.iloc[train_idx]
    test_data = new_data.iloc[test_idx]
    
    print(model.load_state_dict(origin_sd))
    
    ''' Get optimizer for each KFold
    optimizer = optimizer_list[i]
    optimizer_list[i] = ''
    i += 1
    '''
    '''
    NUM_TRAINING_STEPS = (new_data.shape[0]//BATCH_SIZE + 1) * EPOCH
    scheduler = get_linear_schedule_with_warmup(optimizer, NUM_TRAINING_STEPS//10, NUM_TRAINING_STEPS)
    '''

    """ TRAINING """
    dataset = BonzDataset(train_data.iloc[:,-3:], None)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    model.train()
    for epoch in trange(EPOCH):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        loss_train = 0
        for idx, (a, b, c) in enumerate(dataloader):
            optimizer.zero_grad()
            predict, loss = model(a.to(DEVICE), 
                                  b.to(DEVICE), 
                                  c.to(DEVICE), 
                                  weight_loss=weight_loss)[:2]   # This is L-BERT
            #predict, loss = model(a.to(DEVICE), None, c.to(DEVICE))[:2]   # This is normal BERT

            ''' Using apex fp16 loss 
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            '''

            ''' normal loss '''
            loss.backward()

            #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            #scheduler.step()

            loss_train += loss.item()

        print(f'Epoch: {epoch}, Loss = {loss_train:.2f}')
                
        
    ''' TESTING  ''' 
    model.eval()
    dataset = BonzDataset(test_data.iloc[:,-3:], None)
    dataloader = DataLoader(dataset, batch_size=40)

    for idx, (a, b, c) in enumerate(dataloader):
        with torch.no_grad():
            predict = model(a.to(DEVICE), b.to(DEVICE))[0] # This is L-BERT
            #predict = model(a.to(DEVICE), None)[0] # This is normal BERT
        last_predict.extend(predict.detach().cpu().numpy().tolist())
        
        
        
last_predict_ = torch.tensor(last_predict)
last_predict_ = torch.softmax(last_predict_, 1)
y_predict = torch.argmax(last_predict_, 1)
y_true = np.asarray(list(new_data.labels))

for i, asp in enumerate(ASPECT_NAMES):
    print(f'{asp}:\n{classification_report(y_true[:,i], y_predict[:,i])}')
    
    
for i, asp in enumerate(ASPECT_NAMES):
    print(f'{asp}:\t{precision_score(y_true[:,i], y_predict[:,i], average="macro"):.2f}\t\
{recall_score(y_true[:,i], y_predict[:,i], average="macro"):.2f}\t\
{f1_score(y_true[:,i], y_predict[:,i], average="macro"):.2f}\t\
{accuracy_score(y_true[:,i], y_predict[:,i]):.2f}')

