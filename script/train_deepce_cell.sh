#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python ../DeepCE/main_deepce.py --drug_file "../DeepCE/data/drugs_smiles.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_train_cell.csv" \
--dev_file "../DeepCE/data/signature_dev_cell.csv" --test_file "../DeepCE/data/signature_test_cell.csv" \
--dropout 0.1 --batch_size 16 --max_epoch 100  > ../DeepCE/output/cellwise_output.txt
