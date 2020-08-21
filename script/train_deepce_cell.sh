#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 python ../DeepCE/main_deepce.py --drug_file "../DeepCE/data/drugs_smiles.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_train_cell_2.csv" \
--dev_file "../DeepCE/data/signature_dev_cell_2.csv" --test_file "../DeepCE/data/signature_test_cell_2.csv" \
--dropout 0.1 --batch_size 16 --max_epoch 800 # > ../DeepCE/output/cellwise_output_ran5.txt
