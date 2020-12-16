'''
stratified split based on split col
'''
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
    total_data.to_csv('signature_total.csv')
    # split_list = list(set(total_data[split_col]))
    # ['A549', 'MCF7', 'HCC515', 'HEPG2', 'HS578T', 'PC3', 'SKBR3', 'MDAMB231', 'JURKAT', 'A375', 'BT20', 'HELA', 'HT29', 'HA1E', 'YAPC']
    #split_list = ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC']
    split_list = ['A549', 'MCF7', 'HCC515', 'HEPG2', 'HS578T', 'PC3', 'SKBR3', 'MDAMB231', 'JURKAT', 'A375', 'BT20', 'HELA', 'HT29', 'HA1E', 'YAPC']
    random.Random(1111).shuffle(split_list)

    split_variable_length = len(split_list)
    split_variable_train_dev, split_variable_test = split_list[split_variable_length//5:], split_list[:split_variable_length//5]

    split_variable_test = ['JURKAT', 'A375', 'HT29']
    print(split_variable_test)
    split_variable_train_dev_length = len(split_variable_train_dev)
    split_variable_train, split_variable_dev = split_variable_train_dev[split_variable_train_dev_length//5:], split_variable_train_dev[:split_variable_train_dev_length//5]
    split_variable_dev = ['BT20', 'HA1E']
    split_variable_train = list(set(split_list) - set(split_variable_test)-set(split_variable_dev))
    print(split_variable_dev)

    final_train_data = total_data[total_data[split_col].isin(split_variable_train)]
    final_test_data = total_data[total_data[split_col].isin(split_variable_test)]
    final_dev_data = total_data[total_data[split_col].isin(split_variable_dev)]

    final_train_data.to_csv(args.new_train_data, index=False)
    final_test_data.to_csv(args.new_test_data,index=False)
    final_dev_data.to_csv(args.new_dev_data,index=False)
