import pandas as pd

from config import *
from util import *


def read_data():
    # train_data size: 2408 x 89 (35 states + 18 features x 3 days)
    # test_data size: 997 x 88 (without last day's positive rate)
    train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values
    train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

    return train_data, valid_data, test_data


def select_feat(train_data, valid_data, test_data):
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]  # get tested_positive
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data  # get features

    feat_column = [i for i in range(raw_x_train.shape[1]) if i == 0 or i >= 35]  # ignore state features

    return raw_x_train[:, feat_column], raw_x_valid[:, feat_column], raw_x_test[:, feat_column], y_train, y_valid
