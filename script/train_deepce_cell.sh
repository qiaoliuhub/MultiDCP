#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 python ../DeepCE/main_deepce.py --drug_file "../DeepCE/data/drugs_smiles.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_train_cell_3.csv" \
--dev_file "../DeepCE/data/signature_dev_cell_3.csv" --test_file "../DeepCE/data/signature_test_cell_3.csv" \
--dropout 0.1 --batch_size 64 --max_epoch 800 --unfreeze_steps 0,0,0,0 # > ../DeepCE/output/cellwise_output_ran5.txt
