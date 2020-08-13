#!/usr/bin/env bash

python reorganize_train_test.py --train_data signature_train.csv --test_data signature_test.csv \
--dev_data signature_dev.csv --split_variable cell_id --new_train_data signature_train_cell_3.csv \
--new_test_data signature_test_cell_3.csv --new_dev_data signature_dev_cell_3.csv
