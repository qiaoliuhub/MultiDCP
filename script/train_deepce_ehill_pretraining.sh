#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../DeepCE/pretrain_ehill_deepce.py --drug_file "../DeepCE/data/drug_smiles_new.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_train_pretraining_ehill_unit.csv" \
--dev_file "../DeepCE/data/signature_dev_pretraining_ehill_unit.csv" --test_file "../DeepCE/data/signature_test_pretraining_ehill_unit.csv" \
--dropout 0.1 --batch_size 64 --max_epoch 100 --unfreeze_steps 0,0,0,0 --all_cells "../DeepCE/data/pretrain_cell_list_ehill.p" \
--cell_ge_file "../DeepCE/data/gene_expression_combat_norm_978" # > ../DeepCE/output/cellwise_output_ran5.txt
