from typing import List, Union
from classifier import Classifier

import json
import os
import numpy as np
import pandas as pd
import hazm



class LinearClassifier(Classifier):
    def __init__(self, args):
        super().__init__()
        self.DATA_PATH = args.data_path
        self.stopwords = self.read_stopwords()
        self.lemmatizer = hazm.Lemmatizer()

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
        return stopwords

    def normalize(self, mesras: Union[str, List[str]]) -> str:
        '''
        clean, lemmatize and remove stopwords
        '''
        mesras = [mesras] if not isinstance(mesras, list) else mesras
        mesras = [' '.join([self.lemmatizer.lemmatize(word) for word in self.clean(mesra).split() if word not in self.stopwords]) for mesra in mesras]
        return ' '.join(mesras) # concat mesras to form a single string


    def read_data(self, file_name: str) -> pd.DataFrame:
        '''
        reads poems from json file and returns them (cleaned) in a dataframe
        '''
        lst_dict = []
        with open(os.path.join(self.DATA_PATH, file_name), 'r', encoding='utf-8') as f:
            lst_dict = json.load(f)

        df = pd.DataFrame(lst_dict)
        df['poem_clean'] = df['poem'].apply(lambda x: self.normalize(x)) # clean and concatenate mesras
        return df


    def train(self, args):
        pass

    def test(self, args):
        pass
