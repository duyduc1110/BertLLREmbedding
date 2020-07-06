import torch
import pandas as pd
import logging
import time
import torch.nn as nn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(1)


from transformers import BertConfig, BertTokenizerFast, BertModel, AdamW
from torch.utils.data import Dataset, DataLoader
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

PRE_TRAINED = 'bert-base-uncased'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class BonzDataset(Dataset):
    def __init__(self, x, y=None):
        super(BonzDataset, self).__init__()
        self.x = x
        self.y = y
        self.tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED)

    def __getitem__(self, i):
        sen = self.x[i]
        encoded = self.tokenizer.encode(sen)
        encoded = pad_sequences([encoded], maxlen=512, padding='post')
        if self.y is None:
            return torch.LongTensor(encoded[0])
        else:
            return torch.LongTensor(encoded[0]), torch.FloatTensor([self.y[i]])

    def __len__(self):
        return self.x.size


class BertBonz(BertModel):
    def __init__(self, config):
        super(BertBonz, self).__init__(config)
        self.config = config
        self.embeddings.add_module('llr_embeddings', nn.Embedding(3, 768, 0))
        self.classifier = nn.Linear(768, 1)
        self.activation = nn.Sigmoid()
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
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if llr_ids is None:
            llr_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        inputs_embeds = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        llr_embeddings = self.embeddings.llr_embeddings(llr_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + llr_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)

        # BERT ENCODER
        encoder_outputs = self.encoder(
            embeddings,
            attention_mask=None,
            head_mask=[None] * 12,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=self.config.output_attentions
        )
        sequence_output = encoder_outputs[0]

        # CLASSIFIER
        CLS_token = sequence_output[:, 0]
        predict = self.activation(self.classifier(CLS_token))

        if labels is not None:
            loss = self.loss_fn(predict, labels)
            outputs = (predict, loss, CLS_token, sequence_output) + encoder_outputs[
                                                                    1:]  # add hidden_states and attentions if they are here
        else:
            outputs = (predict, CLS_token, sequence_output) + encoder_outputs[1:]
        return outputs

    def load_pretrained_weight(self):
        sd = self.state_dict()
        sd_bert_pretrained = BertModel.from_pretrained(PRE_TRAINED).state_dict()
        for k in sd_bert_pretrained.keys():
            if k in sd.keys():
                sd[k] = sd_bert_pretrained[k]
        self.load_state_dict(sd)
        print('Successfully load pre-trained weights')

    def fit(self,
            optimizer=None,
            loss=None):
        self.optimizer = optimizer(self.parameters(), 2e-5)
        self.loss_fn = loss

    def train_(self,
               inputs=None,
               labels=None,
               epochs=None,
               batch_size=None,
               num_worker=0):
        self.to(DEVICE)
        self.train()

        for epoch in range(epochs):
            my_dataset = BonzDataset(inputs, labels)
            dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)

            s = time.time()
            loss_train = 0
            predicts = []
            y_true = []

            for x, y in dataloader:
                self.optimizer.zero_grad()
                outputs = self(input_ids=x.to(DEVICE), labels=y.to(DEVICE))
                loss = outputs[1]
                loss.backward()
                self.optimizer.step()
                loss_train += loss.item()

                pred = outputs[0].detach().cpu().numpy()
                pred = pred > 0.5
                predicts.extend(pred.squeeze(-1).tolist())
                y_true.extend(y.numpy().squeeze(-1).tolist())

            print(f'Finish epoch {epoch + 1}, loss = {loss_train:.2f}, running time {time.time() - s:.2f}')
            print(classification_report(y_true, predicts))


def get_data():
    data = pd.read_csv('data.csv', sep='\t', encoding='utf8', error_bad_lines=False)
    data['sentiment'] = [1 if rating > 3 else 0 for rating in data['Review\'s Star Rating'].values]
    data['num_words'] = [len(content.split(' ')) for content in data['Review\'s Content'].values]

    return data


if __name__ == '__main__':
    data = get_data()
    data = data.sample(frac=1).reset_index(drop=True).iloc[:10000]

    # Config model
    config = BertConfig(num_labels=1)

    # Create model
    model = BertBonz(config)
    model.load_pretrained_weight()
    model.fit(optimizer=AdamW, loss=nn.BCELoss())

    # Train model
    model.train_(inputs=data['Review\'s Content'].values,
                 labels=data.sentiment.values,
                 epochs=3,
                 batch_size=6,
                 num_worker=0)

    torch.save(model.state_dict(), './weights/llr_embed.pth')
