from classifier import Classifier
import json
import math
import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from multiprocess import set_start_method
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification  # Has MLM head
from transformers import (AutoConfig, AutoTokenizer,
                          DataCollatorWithPadding, Trainer,
                          TrainingArguments, AdamW)


class PoetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class LMClassifier(Classifier):
    def __init__(self, args):
        super().__init__()

        self.MODEL_NAME = args.transformer_model_name
        self.MODEL_DIR = args.models_dir
        self.TRAIN_DATA_PATH = args.data_path
        self.DEVICE = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.config = AutoConfig.from_pretrained(
            self.MODEL_DIR if args.load_model else self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_DIR if args.load_model else self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_DIR if args.load_model else self.MODEL_NAME, num_labels=10).to(self.DEVICE)

        self.train_data = {}
        self.eval_data = {}
        if args.train:
            self.load_data(self.TRAIN_DATA_PATH)
        # Only fine tune the MLM head:
        if args.freeze_encoder:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

    def load(self, path: str):
        pass

    def load_data(self, path: str):
        f_train = open(path + 'train.json', "r")
        self.train_data = json.loads(f_train.read())
        f_train.close()

        f_eval = open(path + 'eval.json', "r")
        self.eval_data = json.loads(f_eval.read())
        f_eval.close()

    def save(self, dir_path: str = None):
        dir_path = dir_path or self.MODEL_DIR
        self.config.save_pretrained(dir_path)
        self.tokenizer.save_pretrained(dir_path)
        self.model.save_pretrained(dir_path)

    def create_dataset(self):

        def in_one_string(poem):
            ss = ''
            for i in range(len(poem) - 1):
                ss += (poem[i] + ' ، ')
            ss += (poem[len(poem) - 1])
            return ss

        def read_data(data):
            texts = []
            labels = []
            for i in range(len(data)):
                labels.append(data[i]['poet'])
                texts.append(in_one_string(data[i]['poem']))

            return texts, labels

        train_texts, train_labels = read_data(self.train_data)
        eval_texts, eval_labels = read_data(self.eval_data)
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        eval_encodings = self.tokenizer(eval_texts, truncation=True, padding=True)
        self.train_dataset = PoetDataset(train_encodings, train_labels)
        self.eval_dataset = PoetDataset(eval_encodings, eval_labels)

        return self.train_dataset, self.eval_dataset

    def classify(self, sent: str):
        pass

    def evaluate(self, model, dataloader):
        score = 0
        total_num = 0
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            X1 = batch['input_ids'].to(self.DEVICE)
            y1 = batch['labels'].to(self.DEVICE)
            pred = model(X1, attention_mask=batch['attention_mask'].to(self.DEVICE))[0]
            pred_label = pred.argmax(dim=1)
            batch_score = sum(pred_label == y1)
            score += batch_score.item()
            total_num += X1.shape[0]
        return (score / total_num)

    def train(self, args):
        train_dataset, eval_dataset = self.create_dataset()
        print('dataset creation done')

        self.model.train()

        data_collator = DataCollatorWithPadding(self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=args.lm_batch_size, collate_fn=data_collator, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.lm_batch_size, collate_fn=data_collator,
                                     shuffle=False)
        optim = AdamW(self.model.parameters(), lr=5e-5)
        print("beginning training")
        for epoch in range(3):
            self.model.train()
            print(epoch)
            print("#####")
            for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.DEVICE)
                attention_mask = batch['attention_mask'].to(self.DEVICE)
                labels = batch['labels'].to(self.DEVICE)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optim.step()
        self.model.eval()
        print("accuracy: ", self.evaluate(self.model, eval_dataloader))

    def test(self, args):
        pass
