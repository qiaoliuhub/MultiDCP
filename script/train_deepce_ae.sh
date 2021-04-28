#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python ../DeepCE/deepce_ae.py --drug_file "../DeepCE/data/all_drugs_l1000.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_train_cell_4_v6.csv" \
--dev_file "../DeepCE/data/signature_dev_cell_4.csv" --test_file "../DeepCE/data/AMPAD_unit_test_dataset_1000.csv" \
--dropout 0.3 --batch_size 64 --max_epoch 500 --unfreeze_steps 0,0,0,0 \
--ae_input_file "../DeepCE/data/gene_expression_combat_norm_978_split4" \
--ae_label_file "../DeepCE/data/gene_expression_combat_norm_978_split4" \
--predicted_result_for_testset "../DeepCE/data/AMPAD_data/low_quality_perturbedGX_v6_unit.csv" \
--hidden_repr_result_for_testset "../DeepCE/data/AMPAD_data/AMPAD_perturbedGX_v6_unit.csv" \
--cell_ge_file "../DeepCE/data/adjusted_ccle_tcga_ad_tpm_log2.csv" \
--all_cells "../DeepCE/data/ccle_tcga_ad_cells.p" \
# > ../DeepCE/output/cellwise_output_ran5.txt
