# train
python train.py --exp_name mvtec_grid_no_mask_base --dataset_name grid      --num_epochs 100000 --max_size 224 --device_ids 7 --use_masks 0 --use_kornia_augm 1 --prob_augm 0.5

python train.py --exp_name mvtec_grid_mask_base    --dataset_name grid      --num_epochs 100000 --max_size 224 --device_ids 4 --use_masks 1 --use_kornia_augm 1 --prob_augm 0.5

python train.py --exp_name mvtec_grid_mask_aug1    --dataset_name grid      --num_epochs 100000 --max_size 224 --device_ids 4 --use_masks 1 --use_kornia_augm 1 --prob_augm 0.5

python train.py --exp_name mvtec_grid_mask_aug1_noaug2    --dataset_name grid      --num_epochs 100000 --max_size 224 --device_ids 6 --use_masks 1 --use_kornia_augm 1 --prob_augm 0.5

python train.py --exp_name mvtec_grid_no_mask_aug4cutpaste --dataset_name grid      --num_epochs 100000 --max_size 224 --device_ids 6 --use_masks 0 --use_kornia_augm 1 --prob_augm 0.5
python train.py --exp_name mvtec_carpet_no_mask_aug4cutpaste --dataset_name carpet      --num_epochs 100000 --max_size 224 --device_ids 5 --use_masks 0 --use_kornia_augm 1 --prob_augm 0.5
python train.py --exp_name mvtec_transistor_no_mask_aug4cutpaste --dataset_name transistor      --num_epochs 100000 --max_size 224 --device_ids 4 --use_masks 0 --use_kornia_augm 1 --prob_augm 0.5

python train.py --exp_name mvtec_carpet_mask_aug1 --dataset_name carpet  --num_epochs 100000 --max_size 224 --device_ids 5 --use_masks 1 --use_kornia_augm 1 --prob_augm 0.5 --use_read_augm 1


python train.py --exp_name mvtec_hazelnut_mask_base --dataset_name hazelnut --num_epochs 100000 --max_size 224 --device_ids 1 --use_masks 1 --use_kornia_augm 1 --prob_augm 0.5
python train.py --exp_name mvtec_hazelnut_mask_aug1_noaug2 --dataset_name hazelnut --num_epochs 100000 --max_size 224 --device_ids 7 --use_masks 1 --use_kornia_augm 1 --prob_augm 0.5 --continue_train 1 --which_epoch 45000

# test
python test.py  --exp_name mvtec_grid_no_mask_aug4cutpaste --dataset_name grid --which_epoch 100000 --max_size 224 --device_ids 0 --use_masks 1 --num_generated 1000


# evaluate
python evaluate.py --exp_name mvtec_grid_no_mask_aug4cutpaste --epoch 00100000 --sifid_all_layers False

