weight_recon=$1
weight_mask=$2
learning_rate=$3
p_drop=$4
p_flip=$5
p_jitter=$6
batch_size=$7
num_epochs=$8

path_load='/home/users/slahlali/diod_3d/tmp/KITTI-burnin_100.0_1.0_0.1_0.3_3/epoch_299.ckpt'

CUDA_VISIBLE_DEVICES=0,1,2,4 python trainKITTI_ts_3D.py \
--data_path /home/users/slahlali/diod_3d/data/KITTI/data \
--model_dir ./tmp \
--sample_dir ./samples \
--proj_name KITTI \
--exp_name KITTI_"$weight_recon"_"$weight_mask"_"$learning_rate"_"$p_drop"_"$p_flip"_"$p_jitter" \
--weight_recon $weight_recon \
--weight_mask $weight_mask \
--start_teacher 500 \
--learning_rate $learning_rate \
--p-drop $p_drop \
--p-flip $p_flip \
--p-jitter $p_jitter \
--batch_size $batch_size \
--num_epochs $num_epochs






weight_recon=1.0
weight_mask=2.0
p_drop=0.1
weight_reg=0.4

path_load='/home/users/slahlali/diod_3d/tmp/KITTI-burnin_100.0_1.0_0.1_0.3_3/epoch_299.ckpt'
CUDA_VISIBLE_DEVICES=0,1,2,3 python trainKITTI_ts_2D_3D.py \
--data_path /home/users/slahlali/diod_3d/data/KITTI/data \
--model_dir ./tmp \
--sample_dir ./samples \
--proj_name KITTI \
--exp_name KITTI-ts_2D_3D \
--weight_recon $weight_recon \
--weight_mask $weight_mask \
--start_teacher 0 \
--learning_rate 0.0005 \
--p-drop $p_drop \
--p-flip 0.0 \
--p-crop 0.0 \
--p-jitter 0.0 \
--batch_size 4 \
--num_epochs 100 \
--weight_reg $weight_reg \
--path_load $path_load