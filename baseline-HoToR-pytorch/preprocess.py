#!/usr/bin/env python
# coding=utf-8
import os
import gzip
import json
import math
import random
import pickle
import pprint
import argparse

import numpy as np
import pandas as pd


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError


class MovieLens1M(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'ratings.dat')

    def load(self):
        # Load data
        df = pd.read_csv(self.fpath,
                         sep='::',
                         engine='python',
                         names=['user', 'item', 'rate', 'time'])
        return df


class MovieLens20M(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'ratings.csv')

    def load(self):
        df = pd.read_csv(self.fpath,
                         sep=',',
                         names=['user', 'item', 'rate', 'time'],
                         usecols=['user', 'item', 'time'],
                         skiprows=1)
        return df


class MovieLens100K(DatasetLoader):
    def __init__(self, data_dir, dataset_num):
        self.train_fpath = os.path.join(data_dir, 'u' + str(dataset_num) + '.base')
        self.test_fpath = os.path.join(data_dir, 'u' + str(dataset_num) + '.test')

    def load(self):
        train_df = pd.read_csv(self.train_fpath,
                               sep='\t',
                               names=['user', 'item', 'rate', 'time'],
                               usecols=['user', 'item', 'rate'])
        test_df = pd.read_csv(self.test_fpath,
                              sep='\t',
                              names=['user', 'item', 'rate', 'time'],
                              usecols=['user', 'item', 'rate'])
        # 测试集只取4/5分为推荐目标
        test_df = test_df[test_df['rate'] >= 4].reset_index(drop=True)

        train_df['user'] -= 1
        train_df['item'] -= 1
        test_df['user'] -= 1
        test_df['item'] -= 1

        return train_df, test_df


class BookCrossing(DatasetLoader):
    def __init__(self, data_dir, dataset_num):
        self.train_fpath = os.path.join(data_dir, 'copy' + str(dataset_num) + '.train')
        self.test_fpath = os.path.join(data_dir, 'copy' + str(dataset_num) + '.test5')

    def load(self):
        train_df = pd.read_csv(self.train_fpath,
                               sep=' ',
                               names=['user', 'item', 'rate'],
                               usecols=['user', 'item', 'rate'])
        test_df = pd.read_csv(self.test_fpath,
                              sep=' ',
                              names=['user', 'item', 'rate'],
                              usecols=['user', 'item', 'rate'])
        train_df['user'] -= 1
        train_df['item'] -= 1
        test_df['user'] -= 1
        test_df['item'] -= 1

        return train_df, test_df


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict


def create_user_list(df, user_size, isTrain=True):
    user_list = [list() for u in range(user_size)]
    for row in df.itertuples():
        user_list[row.user].append((row.item, row.rate) if isTrain else row.item)

    return user_list


def split_train_test(train_df, test_df, user_size):
    train_df.to_csv(os.path.join("preprocessed", args.dataset, "train.base"), header=None, index=None)
    test_df.to_csv(os.path.join("preprocessed", args.dataset, "test.base"), header=None, index=None)

    test_user_list = create_user_list(test_df, user_size, isTrain=False)
    train_user_list = create_user_list(train_df, user_size, isTrain=True)

    return train_user_list, test_user_list


def create_pair(user_list):
    click_pair, buy_pair = [], []
    for user, item_list in enumerate(user_list):
        for item, rate in item_list:
            click_pair.append((user, item, rate)) if rate < 5 else buy_pair.append((user, item, rate))
    return click_pair, buy_pair


def main(args):
    if args.dataset == 'ml-1m':
        df = MovieLens1M(args.data_dir).load()
    elif args.dataset == 'ml-20m':
        df = MovieLens20M(args.data_dir).load()
    elif args.dataset == 'ml-100k':
        train_df, test_df = MovieLens100K(args.data_dir, args.dataset_num).load()
    elif args.dataset == 'BookCrossing':
        train_df, test_df = BookCrossing(args.data_dir, args.dataset_num).load()
    else:
        raise NotImplementedError

    dirname = os.path.join("preprocessed", args.dataset)
    os.makedirs(dirname, exist_ok=True)

    user_size = max(train_df['user'].unique()) + 1
    item_size = max(train_df['item'].unique()) + 1

    train_user_list, test_user_list = split_train_test(train_df, test_df, user_size)
    print('Complete spliting items for training and testing')

    click_train_pair, buy_train_pair = create_pair(train_user_list)
    print('Complete creating pair')

    dataset = {'user_size': user_size, 'item_size': item_size,
               'train_user_list': train_user_list, 'test_user_list': test_user_list,
               'click_train_pair': click_train_pair, 'buy_train_pair': buy_train_pair}
    dirname = os.path.dirname(os.path.abspath(args.output_data))
    os.makedirs(dirname, exist_ok=True)
    with open(args.output_data, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=['ml-1m', 'ml-20m', 'ml-100k', 'BookCrossing', 'ML-10M', 'Netflix'],
                        default='BookCrossing')
    parser.add_argument('--data_dir',
                        type=str,
                        default=os.path.join('../', 'dataset', 'BookCrossing-Rating2Ranking'),
                        help="File path for raw data")
    parser.add_argument('--dataset_num',
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        default=1,
                        help="File path for raw data")
    parser.add_argument('--output_data',
                        type=str,
                        default=os.path.join('preprocessed', 'BookCrossing', 'BookCrossing.pickle'),
                        help="File path for preprocessed data")
    args = parser.parse_args()
    main(args)
