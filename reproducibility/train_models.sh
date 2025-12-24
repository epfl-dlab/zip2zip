torchrun --standalone --nproc_per_node=2 -m train --config="phi_3_14b_train_cfg.yaml"
torchrun --standalone --nproc_per_node=2 -m train --config="phi_3.5_4b_train_cfg.yaml"
