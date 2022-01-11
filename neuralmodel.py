from classifier import Classifier
import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import os
from transformers import AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from tqdm import tqdm


class LSTM(nn.Module):

    def __init__(self, vocab_size, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, 256)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=dimension,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(2*dimension, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, text, text_len):
        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(
            text_emb, text_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)

        out = self.drop(out_reduced)
        out = nn.functional.relu(self.fc1(out))
        out = self.fc2(self.drop(out))

        return out


class NeuralClassifier(Classifier):
    def __init__(self, args):
        super().__init__()
        self.MODEL_NAME = args.transformer_model_name
        self.MODELS_DIR = args.models_dir
        self.DATA_PATH = args.data_path
        self.DEVICE = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_DIR if args.load_model else self.MODEL_NAME)
        self.read_data()
        self.model = LSTM(len(self.text_field.vocab))
        if args.load_model:
            self.load(self.DATA_PATH)
        self.model.to(self.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.neural_lr)

    def load(self, path: str):
        self.model = torch.load(os.path.join(
            path, 'lstm.pt'), map_location=self.DEVICE)

    def save(self, path: str):
        torch.save(self.model, os.path.join(path, 'lstm.pt'))

    def classify(self, sent: str):
        pass #TODO

    def train(self, args):
        # initialize running values
        running_loss = 0.0
        valid_running_loss = 0.0
        device = self.DEVICE
        criterion = nn.CrossEntropyLoss()
        eval_freq = args.neural_eval_freq

        # training loop
        self.model.train()
        for epoch in range(args.neural_epochs):
            for ((text, text_len), labels), _ in tqdm(self.train_loader):
                labels = labels.long().to(device)
                text = text.to(device)
                # text_len = text_len.to(device)
                output = self.model(text, text_len)

                loss = criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update running values
                running_loss += loss.item()

            # evaluation step
            if epoch % eval_freq == eval_freq-1:
                correct, total = 0, 0
                self.model.eval()
                with torch.no_grad():
                    for ((text, text_len), labels), _ in self.val_loader:
                        labels = labels.long().to(device)
                        text = text.to(device)
                        text_len = text_len.to(device)
                        output = self.model(text, text_len)
                        predictions = output.argmax(dim=1)
                        total += len(labels)
                        correct += (predictions == labels).sum().cpu().item()

                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = (
                    running_loss / len(self.train_loader)) / eval_freq
                average_valid_loss = valid_running_loss / len(self.val_loader)
                valid_acc = correct/total

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                correct, total = 0, 0
                self.model.train()

                # print progress
                print(
                    f'Epoch [{epoch+1}/{args.neural_epochs}], Train Loss: {average_train_loss:.2f}, Valid Loss: {average_valid_loss:.2f}, Valid Accuracy: {valid_acc:.2f}')
                self.save(self.DATA_PATH)  # save every epoch
        print('Training finished')

    def test(self, args):
        pass # TODO

    def read_stopwords(self, file_name: str = 'stop_words.txt'):
        stopwords = []
        with open(os.path.join(self.DATA_PATH, file_name), 'r', encoding='utf-8') as f:
            stopwords = f.readlines()
        return [word.strip() for word in stopwords]

    def read_data(self):
        '''
        reads poems from json file and returns them (cleaned) in a dataframe
        '''
        for file_name in ['train', 'eval', 'test']:
            lst_dict = []
            with open(os.path.join(self.DATA_PATH, file_name+'.json'), 'r', encoding='utf-8') as f:
                lst_dict = json.load(f)

            df = pd.DataFrame(lst_dict)
            max_words = 400 #256?
            df['label'] = df['poet']
            df['text'] = df['poem'].apply(lambda mesras: ' '.join(
                '، '.join([self.clean(x) for x in mesras]).split(' ')[:max_words]))
            df = df.reindex(columns=['text', 'label'])
            df.to_csv(os.path.join(self.DATA_PATH,
                      file_name+'.csv'), index=False)

        # Fields
        label_field = Field(sequential=False, use_vocab=False,
                            batch_first=True, dtype=torch.float)
        self.text_field = Field(tokenize=self.tokenizer.tokenize, include_lengths=True, batch_first=True, stop_words=[
            self.tokenizer.tokenize(x)[0] for x in self.read_stopwords()])
        # text_field = Field(include_lengths=True, batch_first=True)
        fields = [('text', self.text_field), ('label', label_field)]

        # TabularDataset
        train, val, test = TabularDataset.splits(
            path=self.DATA_PATH, train='train.csv', validation='eval.csv', test='test.csv', format='CSV', fields=fields, skip_header=True)
        # Iterators
        self.train_loader = BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.text),
                                           device=self.DEVICE, sort=True, sort_within_batch=True)
        self.val_loader = BucketIterator(val, batch_size=32, sort_key=lambda x: len(x.text),
                                         device=self.DEVICE, sort=True, sort_within_batch=True)
        self.test_loader = BucketIterator(test, batch_size=32, sort_key=lambda x: len(x.text),
                                          device=self.DEVICE, sort=True, sort_within_batch=True)

        # Vocabulary
        self.text_field.build_vocab(train, min_freq=3)
