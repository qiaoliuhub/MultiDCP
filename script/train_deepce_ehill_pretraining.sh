#!/usr/bin/env bash

<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=0 python ../DeepCE/pretrain_ehill_deepce.py --drug_file "../DeepCE/data/drug_smiles_new.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --hill_train_file "../DeepCE/data/signature_train_pretraining_ehill.csv" \
--hill_dev_file "../DeepCE/data/signature_dev_pretraining_ehill.csv" \
--hill_test_file "../DeepCE/data/signature_test_pretraining_ehill.csv" \
=======
CUDA_VISIBLE_DEVICES=5 python ../DeepCE/pretrain_ehill_deepce.py --drug_file "../DeepCE/data/drug_smiles_new.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --hill_train_file "../DeepCE/data/signature_train_pretraining_ehill_unit.csv" \
--hill_dev_file "../DeepCE/data/signature_dev_pretraining_ehill_unit.csv" \
--hill_test_file "../DeepCE/data/signature_test_pretraining_ehill_unit.csv" \
>>>>>>> af33ca5bbbc4b5f625319e743e5d236c7b4f94f7
--train_file "../DeepCE/data/signature_train_cell_3.csv" \
--dev_file "../DeepCE/data/signature_dev_cell_3.csv" \
--test_file "../DeepCE/data/signature_test_cell_3.csv" \
--dropout 0.3 --batch_size 256 --max_epoch 100 --unfreeze_steps 0,0,0,0 \
--all_cells "../DeepCE/data/pretrain_cell_list_ehill.p" \
--cell_ge_file "../DeepCE/data/gene_expression_combat_norm_978" \
--linear_only #> ../DeepCE/output/cellwise_output_ran5.txt
