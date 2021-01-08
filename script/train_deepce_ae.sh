#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 python ../DeepCE/deepce_ae.py --drug_file "../DeepCE/data/drug_smiles_new.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_train_cell_1.csv" \
--dev_file "../DeepCE/data/signature_dev_cell_1.csv" --test_file "../DeepCE/data/signature_test_cell_1.csv" \
--dropout 0.3 --batch_size 64 --max_epoch 800 --unfreeze_steps 0,0,0,0 \
--ae_input_file "../DeepCE/data/gene_expression_combat_norm_978_split1" \
--ae_label_file "../DeepCE/data/gene_expression_combat_norm_978_split1"  \
--predicted_result_for_testset "../DeepCE/data/side_effect/FAERS_offsides_PTs_PredictionDGX.csv" \
--cell_ge_file "../DeepCE/data/gene_expression_combat_norm_978" # > ../DeepCE/output/cellwise_output_ran5.txt
