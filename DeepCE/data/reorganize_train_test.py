#! /usr/bin/env python

import pandas as pd
import argparse
import random

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepCE data respliting')
    parser.add_argument('--train_data')
    parser.add_argument('--test_data')
    parser.add_argument('--dev_data')
    parser.add_argument('--split_variable')
    parser.add_argument('--new_train_data')
    parser.add_argument('--new_test_data')
    parser.add_argument('--new_dev_data')

    args = parser.parse_args()

    train_data = pd.read_csv(args.train_data)
    test_data = pd.read_csv(args.test_data)
    dev_data = pd.read_csv(args.dev_data)
    split_col = args.split_variable

    total_data = pd.concat([train_data, test_data, dev_data])
    split_list = list(set(total_data[split_col]))
    random.shuffle(split_list)

    split_variable_length = len(split_list)
    split_variable_train_dev, split_variable_test = split_list[split_variable_length//5:], split_list[:split_variable_length//5]
    split_variable_train_dev_length = len(split_variable_train_dev)
    split_variable_train, split_variable_dev = split_variable_train_dev[split_variable_train_dev_length//5:], split_variable_train_dev[:split_variable_train_dev_length//5]
    
    final_train_data = total_data[total_data[split_col].isin(split_variable_train)]
    final_test_data = total_data[total_data[split_col].isin(split_variable_test)]
    final_dev_data = total_data[total_data[split_col].isin(split_variable_dev)]

    final_train_data.to_csv(args.new_train_data, index=False)
    final_test_data.to_csv(args.new_test_data,index=False)
    final_dev_data.to_csv(args.new_dev_data,index=False)
