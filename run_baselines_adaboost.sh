
data_types=("all_0426") 
available_gpus=(4 5 0 1 2 3)  # 这里假设你有5个GPU: 0, 1, 2, 3, 4
num_gpus=${#available_gpus[@]}
if_rest_onlys=("True" "False")
if_PCAs=("True" "False")


# 循环遍历test_user_list_idx从0到4
# for test_user_list_idx in {0..4}; do
for data_type in "${data_types[@]}"; do
    for if_rest_only in "${if_rest_onlys[@]}"; do
        for if_PCA in "${if_PCAs[@]}"; do
            log_folder="/home/mmWave_group/EasyCog/logs/Cog_Baselines/Handfeat_Adaboost_validv2/${data_type}_restonly_${if_rest_only}_PCA_${if_PCA}"
            log_testset_folder="${log_folder}"
            mkdir -p "$log_testset_folder"
            # 计算使用哪个GPU (循环使用可用GPU)
            gpu_index=$((test_user_list_idx % num_gpus))
            gpu_id=${available_gpus[$gpu_index]}

            echo "test_user_list_idx: ${test_user_list_idx}, gpu_id: ${gpu_id}"
            # 创建日志文件名
            log_filename="log_adaboost_five_folds.out"
            log_file="${log_testset_folder}/${log_filename}"
            
            echo "启动测试集 ${test_user_list_idx} 在 GPU ${gpu_id} 上..."

            nohup python DL_pipeline/Adaboost_baseline_five_fold_validation.py \
                --data_type ${data_type} \
                --log_file "$log_file" \
                --if_rest_only ${if_rest_only} \
                --if_mean "True" \
                --if_PCA ${if_PCA} \
                &>> "$log_file" &
            
            pid=$!
            echo "[Process ID for test_user_list_idx=${test_user_list_idx} on GPU ${gpu_id}]: $pid" >> "$log_file"
            echo "进程已启动，PID: $pid，日志文件: $log_file"
            
            # 可选：添加短暂延迟，避免同时启动多个进程
            sleep 15m
        done
    done
done

echo "所有测试集已启动！"
