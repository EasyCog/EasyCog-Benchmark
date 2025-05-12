# cfg_file='Ours/Joint_Train/cfg_mocov2_sample_joint_train_classifier_mask_pretrain_CBraMod.yml'
# cfg_file_reg='Ours/Joint_Train/cfg_mocov2_sample_joint_train_regressor_mask_pretrain_RegTransformer.yml'
# log_folder="../EasyCog/logs/Ours/moco_sample_visual_alignment_task_attn_randomorder_mixup_nopinknoise/"

# cfg_file='Ours/Ablation/No_Visual_Alignment/cfg_mocov2_sample_joint_train_classifier_mask_pretrain_CBraMod.yml'
# cfg_file_reg='Ours/Ablation/No_Visual_Alignment/cfg_mocov2_sample_joint_train_regressor_mask_pretrain_RegTransformer.yml'
# log_folder="../EasyCog/logs/Ours/Ablation/No_Visual_Alignment/"
# cfg_file='Ours/Ablation/No_Contrastive_CLS/cfg_joint_train_classifier_mask_pretrain_CBraMod.yml'
# cfg_file_reg='Ours/Ablation/No_Contrastive_CLS/cfg_joint_train_regressor_mask_pretrain_RegTransformer.yml'
# log_folder="../EasyCog/logs/Ours/Ablation/No_Contrastive_CLS/"

# cfg_file='Ours/Ablation/No_MoCo_No_VA_No_TaskAttn_NoSubjectMixup/cfg_joint_train_classifier_mask_pretrain_CBraMod.yml'
# cfg_file_reg='Ours/Ablation/No_MoCo_No_VA_No_TaskAttn_NoSubjectMixup/cfg_joint_train_regressor_mask_pretrain_RegTransformer.yml'
# log_folder="../EasyCog/logs/Ours/Ablation/No_MoCo_No_VA_No_TaskAttn_NoSubjectMixup/"

cfg_file='Ours/Ablation/Only_Video/cfg_mocov2_sample_joint_train_classifier_mask_pretrain_CBraMod.yml'
cfg_file_reg='Ours/Ablation/Only_Video/cfg_mocov2_sample_joint_train_regressor_mask_pretrain_RegTransformer.yml'
log_folder="../EasyCog/logs/Ours/Ablation/Only_Video/"
mkdir -p "$log_folder"

data_types=("all_0426") # "all_0426")
available_gpus=(3 4 5)  
num_gpus=${#available_gpus[@]}

gpu_list_str=$(IFS=", "; echo "${available_gpus[*]}")
echo "Using GPUs: $gpu_list_str"


for data_type in "${data_types[@]}"; do
    log_folder="${log_folder}/${data_type}"
    record_file="${log_folder}/record_test_valid_user_list_idx.out"
    mkdir -p "$log_folder"
    log_testset_folder="${log_folder}/"
    mkdir -p "$log_testset_folder"
    for test_user_list_idx in {3..5}; do
        gpu_index=$((test_user_list_idx % num_gpus))
        gpu_id=${available_gpus[$gpu_index]}


        valid_user_list_idx=$(( (test_user_list_idx + 1) % 10 ))

        echo "test_user_list_idx: ${test_user_list_idx}, valid_user_list_idx: ${valid_user_list_idx}, gpu_id: ${gpu_id}" >> "$record_file"
        log_filename="log_${data_type}_testset${test_user_list_idx}.out"
        log_file="${log_testset_folder}/${log_filename}"
        
        echo "Start testing: ${test_user_list_idx} on GPU ${gpu_id}..." >> "$record_file"

        nohup python main_validation_save_version.py \
            --data_type ${data_type} \
            --test_user_list_idx ${test_user_list_idx} \
            --valid_user_list_idx ${valid_user_list_idx} \
            --log_file "$log_file" \
            --task_pretrain False \
            --cog_regression True \
            --cfg_file "$cfg_file" \
            --cog_regression_cfg_file "$cfg_file_reg" \
            --joint_cls_reg_train True \
            --gpu ${gpu_id} \
            --wait_time $((0)) \
            --gpu_memory 17 \
            &>> "$log_file" &
        
        pid=$!
        echo "[Process ID for test_user_list_idx=${test_user_list_idx} on GPU ${gpu_id}]: $pid" >> "$log_file"        

        sleep 2
    done
done