#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../DeepCE/deepce_ae.py --drug_file "../DeepCE/data/drug_smiles_new.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_train_dev_cell_1.csv" \
--dev_file "../DeepCE/data/signature_test_cell_1.csv" --test_file "../DeepCE/data/side_effect/SIDER_PTs_0.3_CellLineDGX.csv" \
--dropout 0.3 --batch_size 64 --max_epoch 150 --unfreeze_steps 0,0,0,0 \
--ae_input_file "../DeepCE/data/gene_expression_combat_norm_978_split4" \
--ae_label_file "../DeepCE/data/gene_expression_combat_norm_978_split4" \
--predicted_result_for_testset "../DeepCE/data/side_effect/SIDER_PTs_0.3_new_PredictionDGX_v1.csv" \
--cell_ge_file "../DeepCE/data/gene_expression_combat_norm_978" # > ../DeepCE/output/cellwise_output_ran5.txt
