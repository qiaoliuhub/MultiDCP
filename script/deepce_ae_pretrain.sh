#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python ../DeepCE/deepce_ae_pretrain.py --drug_file "../DeepCE/data/drugs_smiles.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_train_cell_3.csv" \
--dev_file "../DeepCE/data/signature_dev_cell_3.csv" --test_file "../DeepCE/data/signature_test_cell_3.csv" \
--dropout 0.1 --batch_size 64 --max_epoch 800 --unfreeze_steps 0,0,0,0 --ae_input_file \
"../DeepCE/data/gene_expression_combat_norm_978_split3" \
--ae_label_file "../DeepCE/data/gene_expression_combat_norm_978_split3"  \
--cell_ge_file "../DeepCE/data/gene_expression_combat_norm_978" \
--linear_only # > ../DeepCE/output/cellwise_output_ran5.txt
