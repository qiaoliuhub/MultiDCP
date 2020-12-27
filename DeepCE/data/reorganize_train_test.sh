#!/usr/bin/env bash

python reorganize_train_test.py --train_data high_confident_data_train.csv --test_data high_confident_data_test.csv \
--dev_data high_confident_data_dev.csv --split_variable cell_id --new_train_data high_confident_data_train.csv \
--new_test_data high_confident_data_test.csv --new_dev_data high_confident_data_dev.csv
