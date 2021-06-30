#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0 python ../MultiDCP/multidcp_ae_second.py --drug_file "../MultiDCP/data/drugs_smiles.csv" \
--gene_file "../MultiDCP/data/gene_vector.csv"  --train_file "../MultiDCP/data/pert_transcriptom/signature_train_cell_3.csv" \
--dev_file "../MultiDCP/data/pert_transcriptom/signature_dev_cell_3.csv" --test_file "../MultiDCP/data/pert_transcriptom/signature_test_cell_3.csv" \
--dropout 0.3 --batch_size 64 --max_epoch 800 --unfreeze_steps 0,0,0,0 --ae_input_file \
"../MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978_split3" --ae_label_file "../MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978_split3"  --cell_ge_file "../MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978" # > ../MultiDCP/output/cellwise_output_ran5.txt
