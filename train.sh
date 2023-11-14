# 
python train.py --exp_name mvtec_hazelnut_no_mask_1 --dataset_name mvtec --num_epochs 150000 --max_size 330 --device_ids 6 --use_masks 0 --use_kornia_augm 1


python train.py --exp_name mvtec_hazelnut_mask --dataset_name mvtec --num_epochs 150000 --max_size 330 --device_ids 5 --use_masks 1 --use_kornia_augm 1 --prob_augm 0.5

python train.py --exp_name mvtec_hazelnut_all_mask --dataset_name mvtec_all --num_epochs 150000 --max_size 330 --device_ids 6 --use_masks 1 --use_kornia_augm 1 --prob_augm 0.5


python test.py --exp_name mvtec_hazelnut --which_epoch 100000 
