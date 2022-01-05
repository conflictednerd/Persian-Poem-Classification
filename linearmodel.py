import json
import os
import pickle
from typing import List, Union

import hazm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import feature_extraction
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

from classifier import Classifier


class LinearClassifier(Classifier):
    def __init__(self, args):
        super().__init__()
        self.DATA_PATH = args.data_path
        self.MODELS_DIR = args.models_dir
        self.stopwords = self.read_stopwords()
        self.lemmatizer = hazm.Lemmatizer()
        self.vectorizer = feature_extraction.text.TfidfVectorizer(
            max_features=args.linear_max_features,
            ngram_range=(args.linear_ngram_min, args.linear_ngram_max),
            max_df=0.8
        )
        self.vocab, self.inv_vocab = {}, {}
        self.model = LogisticRegression(class_weight='balanced', n_jobs=-1)

        if args.load_model:
            self.load()

    def load(self, path: str = 'linear_model.pkl'):
        with open(os.path.join(self.MODELS_DIR, path), 'rb') as f:
            temp = pickle.load(f)
        self.vectorizer = temp['vectorizer']
        self.vocab = temp['vocab']
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.model = temp['model']

    def save(self, path: str = 'linear_model.pkl'):
        with open(os.path.join(self.MODELS_DIR, path), 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'vocab': self.vocab,
                'model': self.model,
            }, f)

    def classify(self, sent: str) -> dict:
        '''
        Classifies a given document.
        :param str sent: document to be classified
        '''
        sent = self.normalize(sent)
        vec = self.vectorizer.transform([sent])
        prediction = self.model.predict(vec)[0]
        cls_probs = self.model.predict_proba(vec)[0]
        coef = self.model.coef_[prediction]
        markers = [(self.inv_vocab[i], np.abs(coef[i]*vec[0, i]))
                   for i in vec.indices]  # (token, impact)
        markers.sort(key=lambda x: x[1], reverse=True)
        markers = [token for token, impact in markers]

        return {
            'prediction': Classifier.POETS[prediction],
            'probs': cls_probs,
            'markers': markers[:6]
        }

    def train(self, args=None):
        df = self.read_data('train.json')
        # X_train.shape = num. of poems * selected_vocab_size
        X_train = self.vectorizer.fit_transform(df['poem_clean'])
        self.vocab = self.vectorizer.vocabulary_  # a dict
        # will be used in extracting markers
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        # TODO: add feature selection
        self.model.fit(X=X_train, y=df['poet'])
        self.save()

    def test(self, args=None):
        print('Evaluation Report:')
        # TODO: change to test.json
        df = self.read_data('eval.json')
        X_test = self.vectorizer.transform(df['poem_clean'])
        ConfusionMatrixDisplay.from_estimator(
            self.model, X_test, df['poet'],
            normalize='true',
            display_labels=['Hafez', 'Khayyam', 'Ferdousi', 'Moulavi',
                            'Nezami', 'Saadi', 'Parvin', 'Sanaie', 'Vahshi', 'Roudaki'],
        )
        plt.gcf().set_size_inches(10, 10)
        plt.savefig(os.path.join(self.DATA_PATH, 'confusion.png'), dpi=100)
        print(classification_report(df['poet'], self.model.predict(X_test)))

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
        mesras = [' '.join([self.lemmatizer.lemmatize(word).replace('#', '') for word in self.clean(
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
