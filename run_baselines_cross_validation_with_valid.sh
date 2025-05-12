
cfg_file='/home/mmWave_group/EasyCog/DL_pipeline/configs/Baselines/cfg_adaptive_GCNNTransformer_end_to_end.yml'

# 可用的GPU列表，根据你的实际情况修改
# available_gpus=(0 1 2 3 5)  # 这里假设你有5个GPU: 0, 1, 2, 3, 4
# data_types=("clean_0426" "all_0426")
# data_types=("clean_0426") # "all_0426")
data_types=("all_0426") # "all_0426")
available_gpus=(4 5 0 1 2 3)  # 这里假设你有5个GPU: 0, 1, 2, 3, 4
num_gpus=${#available_gpus[@]}



# 循环遍历test_user_list_idx从0到4
# for test_user_list_idx in {0..4}; do
for data_type in "${data_types[@]}"; do
    log_folder="/home/mmWave_group/EasyCog/logs/Cog_Baselines/Adaptive_SpatialTemporal/${data_type}"
    mkdir -p "$log_folder"
    log_testset_folder="${log_folder}/"
    mkdir -p "$log_testset_folder"
    for test_user_list_idx in {0..9}; do
        # 计算使用哪个GPU (循环使用可用GPU)
        gpu_index=$((test_user_list_idx % num_gpus))
        gpu_id=${available_gpus[$gpu_index]}

        # 计算验证集用户列表索引 (循环使用)
        # 当 test_user_list_idx 为 9 时, valid_user_list_idx 计算结果为 (9+1)%10 = 0
        valid_user_list_idx=$(( (test_user_list_idx + 1) % 10 ))

        echo "test_user_list_idx: ${test_user_list_idx}, valid_user_list_idx: ${valid_user_list_idx}, gpu_id: ${gpu_id}"
        # 创建日志文件名
        log_filename="log_adaptive_spatialtemporal_subject_norm_${data_type}_testset${test_user_list_idx}.out"
        log_file="${log_testset_folder}/${log_filename}"
        
        echo "启动测试集 ${test_user_list_idx} 在 GPU ${gpu_id} 上..."
        
        # 启动训练进程
        ### This is an end-to-end training. so we use the same pipeline as task_pretrain
        nohup python main_validation_save_version.py \
            --data_type ${data_type} \
            --test_user_list_idx ${test_user_list_idx} \
            --valid_user_list_idx ${valid_user_list_idx} \
            --log_file "$log_file" \
            --task_pretrain True \
            --cog_regression False \
            --cfg_file "$cfg_file" \
            --joint_cls_reg_train False \
            --gpu ${gpu_id} \
            --wait_time $((0)) \
            &>> "$log_file" &
        
        pid=$!
        echo "[Process ID for test_user_list_idx=${test_user_list_idx} on GPU ${gpu_id}]: $pid" >> "$log_file"
        echo "进程已启动，PID: $pid，日志文件: $log_file"
        
        # 可选：添加短暂延迟，避免同时启动多个进程
        sleep 2
    done
done

echo "所有测试集已启动！"
