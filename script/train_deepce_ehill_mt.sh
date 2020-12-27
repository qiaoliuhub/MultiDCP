#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=6 python ../DeepCE/ehill_deepce_mt.py --drug_file "../DeepCE/data/drug_smiles_new.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --hill_train_file "../DeepCE/data/high_confident_data_train_uniq.csv" \
--hill_dev_file "../DeepCE/data/high_confident_data_dev_uniq.csv" \
--hill_test_file "../DeepCE/data/high_confident_data_test_uniq.csv" \
--train_file "../DeepCE/data/signature_train_cell_2.csv" \
--dev_file "../DeepCE/data/signature_dev_cell_2.csv" \
--test_file "../DeepCE/data/signature_test_cell_2.csv" \
--dropout 0.1 --batch_size 64 --max_epoch 100 --unfreeze_steps 0,0,0,0 \
--all_cells "../DeepCE/data/pretrain_cell_list_ehill.p" \
--ae_input_file "../DeepCE/data/gene_expression_combat_norm_978_split1" \
--ae_label_file "../DeepCE/data/gene_expression_combat_norm_978_split1"  \
--cell_ge_file "../DeepCE/data/gene_expression_combat_norm_978" \
--linear_only # > ../DeepCE/output/cellwise_output_ran5.txt
