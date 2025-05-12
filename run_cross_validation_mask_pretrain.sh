
log_folder="/home/mmWave_group/EasyCog/logs/Ours/Mask_Pretrain/"
cfg_file='/home/mmWave_group/EasyCog/DL_pipeline/configs/Ours/Mask_Pretrain/cfg_CBraMod_Pretrained_asreog_filter3_Mask_Autoencoder_Pretrain.yml'


available_gpus=(0 1 2 3 4 5)  
num_gpus=${#available_gpus[@]}
data_type="all_0426" #"all_0426" or "clean_0426"
mkdir -p "$log_folder"
log_testset_folder="${log_folder}/${data_type}"
mkdir -p "$log_testset_folder"
for test_user_list_idx in {0..9}; do
    gpu_id=${available_gpus[$test_user_list_idx % $num_gpus]}
    
    log_filename="log_Ours_Mask_Pretrain_testset${test_user_list_idx}.out"
    log_file="${log_testset_folder}/${log_filename}"
    
    echo "Starting testing ${test_user_list_idx} on GPU ${gpu_id}..."
    

    nohup python main.py \
        --data_type ${data_type} \
        --test_user_list_idx ${test_user_list_idx} \
        --log_file "$log_file" \
        --task_pretrain True \
        --cog_regression False \
        --cfg_file "$cfg_file" \
        --joint_cls_reg_train False \
        --gpu ${gpu_id} \
        &>> "$log_file" &
    
    pid=$!
    echo "[Process ID for test_user_list_idx=${test_user_list_idx} on GPU ${gpu_id}]: $pid" >> "$log_file"
    
    sleep 2
done

echo "所有测试集已启动！"