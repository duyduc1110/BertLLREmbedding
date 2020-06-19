import torch
import numpy as np
import pandas as pd
import logging
import time
import torch.nn as nn

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, BertTokenizerFast, BertModel, AdamW
from transformers.modeling_bert import BertEmbeddings, BertSelfAttention
from torch.utils.data import Dataset, DataLoader
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import trange, tqdm, tqdm_notebook, tqdm_pandas, tqdm_gui

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

PRE_TRAINED = 'bert-base-uncased'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyDataset(Dataset):
    def __init__(self, x, y=None):
        super(MyDataset, self).__init__()
        self.x = x
        self.y = y
        self.tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED)

    def __getitem__(self, i):
        sen = self.x[i]
        encoded = self.tokenizer.encode(sen)
        encoded = pad_sequences([encoded], maxlen=512, padding='post')
        if self.y is None:
            return torch.FloatTensor(encoded[0])
        else:
            return torch.LongTensor(encoded[0]), torch.FloatTensor([self.y[i]])

    def __len__(self):
        return self.x.size


class NewBert(nn.Module):
    def __init__(self, config):
        super(NewBert, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED)
        self.classifier = nn.Linear(768, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x, y=None):
        loss_fn = nn.BCELoss()
        last_hidden_state = self.bert(x)[0]
        CLS_token = last_hidden_state[:, 0]
        x = self.classifier(CLS_token)
        x = self.activation(x)
        if y is not None:
            loss = loss_fn(x, y)
            return loss, x
        else:
            return x


def get_data():
    data = pd.read_csv('data.csv', sep='\t', encoding='utf8', error_bad_lines=False)
    data['sentiment'] = [1 if rating > 3 else 0 for rating in data['Review\'s Star Rating'].values]
    data['num_words'] = [len(content.split(' ')) for content in data['Review\'s Content'].values]

    return data


def get_model():
    config = BertConfig(num_labels=1)
    model = BertForSequenceClassification(config=config)
    sd = model.state_dict()

    bert = BertModel.from_pretrained(PRE_TRAINED)
    bert_sd = bert.state_dict()

    for key in bert_sd.keys():
        sd['bert.' + key] = bert_sd[key]

    model.load_state_dict(sd)

    return config, model


def train(model=None, epochs=None):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.train()
    model.to(DEVICE)

    for epoch in range(epochs):
        my_dataset = MyDataset(x=data['Review\'s Content'].values, y=data.sentiment.values)
        dataloader = DataLoader(my_dataset, batch_size=4, shuffle=True)
        s = time.time()
        for x, y in dataloader:
            optimizer.zero_grad()
            loss = model(x=x.to(DEVICE), y=y.to(DEVICE))[0]
            loss.backward()
            optimizer.step()
        print('Finish epoch {}, running time {}'.format(epoch + 1, time.time() - s))

    model.eval()
    predicts = []
    y_true = []
    for x, y in dataloader:
        with torch.no_grad():
            predict = model(x=x.to(DEVICE))
        predict = predict.detach().cpu().numpy()
        predict = predict > 0.5
        predicts.extend(predict.tolist())
        y_true.extend(y.numpy().tolist())

    print(classification_report(y_true, predicts))
    return model


def pre_dict(model=None, data=None):
    my_dataset = MyDataset(x=data['Review\'s Content'].values, y=data.sentiment.values)
    dataloader = DataLoader(my_dataset, batch_size=4)

    model.eval()
    predicts = []
    print('Start predicting!!!')
    for x, _ in dataloader:
        with torch.no_grad():
            predict = model(x=x.to(DEVICE))
        predict = predict.detach().cpu().numpy()
        predict = predict > 0.5
        predicts.extend(predict.tolist())

    print(classification_report(data.sentiment.values, predicts))
    return model



if __name__ == '__main__':
    data = get_data()
    #data = data.sample(frac=1).reset_index(drop=True).iloc[:10000]

    # Config model
    config = BertConfig(num_labels=1)
    model = NewBert(config)
    model.to(DEVICE)
    sd = torch.load('weights.h5')
    model.load_state_dict(sd)

    # Train model
    #model = train(model, 3)
    #torch.save(model.state_dict(), 'weights.h5')

    # Predict test data
    pre_dict(model, data)
