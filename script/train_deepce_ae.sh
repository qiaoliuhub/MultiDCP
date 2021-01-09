#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python ../DeepCE/deepce_ae.py --drug_file "../DeepCE/data/drug_smiles_new.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_total_train_cell.csv" \
--dev_file "../DeepCE/data/signature_dev_cell_1.csv" --test_file "../DeepCE/data/side_effect/combined_CellLineDGX.csv" \
--dropout 0.3 --batch_size 64 --max_epoch 1 --unfreeze_steps 0,0,0,0 \
--ae_input_file "../DeepCE/data/gene_expression_combat_norm_978_split4" \
--ae_label_file "../DeepCE/data/gene_expression_combat_norm_978_split4"  \
--predicted_result_for_testset "../DeepCE/data/side_effect/combined_PredictionDGX.csv" \
--cell_ge_file "../DeepCE/data/gene_expression_combat_norm_978" # > ../DeepCE/output/cellwise_output_ran5.txt


# dosage_map = {'1.11 µM':'1.11 um', 
# '40 µM': '40 um', '0.04 um': '0.04 um', '0.37 um': '0.37 um', 
# '100 nM': '0.1 um', '1 µM': '1 um', '0.04 µM': '0.04 um', '10.05 um': '10.0 um', 
# '1 nM': '1 nm', '20 µM': '20 um', '3.33 µM': '3.33 um', '1.11 um': '1.11 um', 
# '5 µM': '5 um', '3.35 um': '3.33 um', '3 µM': '3 um', '100 µM': '100 um', 
# '10 µM': '10.0 um', '50 µM': '50 um', '80 µM': '80 um', '30 µM': '30 um', 
# '90 µM': '90 um', '0.12 um': '0.12 um', '500 nM': '0.5 um', '1.12 um': '1.11um', 
# '10 nM': '10.0 nm', '10.0 um': '10.0 um', '20.0 um': '20.0 um', 
# '3.33 um': '3.33 um', '0.12 µM': '0.12 um', '0.37 µM': '0.37 um'}