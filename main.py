import argparse
from pprint import pprint

from langmodel import LMClassifier
from linearmodel import LinearClassifier
from neuralmodel import NeuralClassifier

'''
A note for ourselves!:
data contains train.json, val.json, test.json
each json file contains a list of samples. each element in the list is a dict with two keys: poem and poet.
poet is the id of the poet to whom the song belongs. poem is a list of strings -> mesras
'''

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lm',
                        help='type of classification model: logistic, neural, lm')
    parser.add_argument('--transformer_model_name',
                        default='HooshvareLab/albert-fa-zwnj-base-v2')
    parser.add_argument('--load_model', action='store_true', default=False,
                        help='use if you want to load the model from file')
    parser.add_argument('--train', action='store_true',
                        default=False, help='use if you want to train the model')
    parser.add_argument('--test', action='store_true', default=False,
                        help='use if you want the model to report performance on held-out test set')
    parser.add_argument('--data_path', default='./data/',
                        help='directory where training data is stored')
    parser.add_argument('--models_dir', default='./models_dir/',
                        help='directory where models are saved to/loaded from')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    pprint(f'Arguments are: {vars(args)}')  # For debugging

    if args.model == 'lm':
        model = LMClassifier(args)
    elif args.model == 'neural':
        model = NeuralClassifier(args)
    else:
        model = LinearClassifier(args)

    if args.train:
        model.train(args)

    if args.test:
        model.test(args)
