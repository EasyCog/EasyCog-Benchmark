data_type="all_0426" # clean_0426 or all_0426
eval_user_set="test"
eval_mode="valid" # test or valid

# log_folder="/home/mmWave_group/EasyCog/logs/Cog_Baselines/Adaptive_SpatialTemporal/${data_type}/"
# log_folder="/home/mmWave_group/EasyCog/logs/Ours/moco_sample_visual_alignment_task_attn_randomorder_mixup/${data_type}"
# log_filename_prefix="log_moco_sample_visual_alignment_pink_noise_subject_norm_${data_type}"
# log_folder="/home/mmWave_group/EasyCog/logs/Cog_Baselines/Adaptive_SpatialTemporal/${data_type}/"
# log_filename_prefix="log_adaptive_spatialtemporal_subject_norm_${data_type}"

# log_folder="/home/mmWave_group/EasyCog/logs/Ours/Ablation/Only_Video/${data_type}/"
# log_filename_prefix="log_${data_type}"

# log_folder="/home/mmWave_group/EasyCog/logs/Ours/Ablation/No_MoCo_No_VA_No_TaskAttn_NoSubjectMixup/${data_type}/"
# log_filename_prefix="log_${data_type}"

log_folder="/home/mmWave_group/EasyCog/logs/Ours/Ablation/No_MoCo_No_VA_No_TaskAttn_NoSubjectMixup/${data_type}/"
log_filename_prefix="log_${data_type}"

minus_last_epoch=25

mkdir -p "$log_folder"
log_eval_folder="${log_folder}/eval_by_${eval_mode}_on_${eval_user_set}set_minus_last_epoch${minus_last_epoch}"
mkdir -p "$log_eval_folder"

log_file="${log_eval_folder}/${log_filename_prefix}.out"
nohup python main_five_fold_test_log_record_v2.py \
    --log_file "$log_filename_prefix" \
    --log_folder "$log_folder" \
    --output_dir "$log_eval_folder" \
    --k_fold 10 \
    --best_epoch_source "${eval_mode}" \
    --eval_user_set "${eval_user_set}" \
    --minus_last_epoch "${minus_last_epoch}" \
    &>> "$log_file" &
echo "[Process ID]: $!" >> "$log_file"


