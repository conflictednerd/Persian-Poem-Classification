import json
import os
from typing import List, Union

import hazm
import numpy as np
import pandas as pd
from sklearn import feature_extraction
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import \
    LogisticRegression  # set class_weight = 'balanced

from classifier import Classifier


class LinearClassifier(Classifier):
    def __init__(self, args):
        super().__init__()
        self.DATA_PATH = args.data_path
        self.stopwords = self.read_stopwords()
        self.lemmatizer = hazm.Lemmatizer()
        self.vectorizer = feature_extraction.text.TfidfVectorizer(
            max_features=args.linear_max_features,
            ngram_range=(args.linear_ngram_min, args.linear_ngram_max)
        )
        self.vocab, self.inv_vocab = {}, {}
        self.model = LogisticRegression(class_weight='balanced', n_jobs=-1)

    def load(self, path: str):
        pass

    def save(self, path: str):
        pass

    def classify(self, sent: str):
        pass

    def read_stopwords(self, file_name: str = 'stop_words.txt'):
        stopwords = []
        with open(os.path.join(self.DATA_PATH, file_name), 'r', encoding='utf-8') as f:
            stopwords = f.readlines()
        return [word.strip() for word in stopwords]

    def normalize(self, mesras: Union[str, List[str]]) -> str:
        '''
        clean, lemmatize and remove stopwords
        '''
        mesras = [mesras] if not isinstance(mesras, list) else mesras
        mesras = [' '.join([self.lemmatizer.lemmatize(word) for word in self.clean(
            mesra).split() if word not in self.stopwords]) for mesra in mesras]
        return ' '.join(mesras)  # concat mesras to form a single string

    def read_data(self, file_name: str) -> pd.DataFrame:
        '''
        reads poems from json file and returns them (cleaned) in a dataframe
        '''
        lst_dict = []
        with open(os.path.join(self.DATA_PATH, file_name), 'r', encoding='utf-8') as f:
            lst_dict = json.load(f)

        df = pd.DataFrame(lst_dict)
        df['poem_clean'] = df['poem'].apply(
            lambda x: self.normalize(x))  # clean and concatenate mesras
        return df

    def train(self, args):
        df = self.read_data(os.path.join(self.DATA_PATH, 'train.json'))
        # X_train.shape = num. of poems * selected_vocab_size
        X_train = self.vectorizer.fit_transform(df['poem_clean'])
        self.vocab = self.vectorizer.vocabulary_  # a dict
        # will be used in extracting markers
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        # TODO: add feature selection
        self.model.fit(X=X_train, y=df['poet'])
        self.save()

    def test(self, args):
        pass
