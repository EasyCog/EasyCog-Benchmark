import os

def extract_best_mae_model_path(log_file_path):
    """
    从日志文件中提取best_mae_model的保存路径
    
    参数:
        log_file_path: 日志文件的路径
        
    返回:
        best_mae_model的路径，如果未找到则返回None
    """
    model_save_string = "Model will be saved at:"
    with open(log_file_path, 'r') as file:
        content = file.read()
        
        lines = content.split('\n')
        for line in lines:  # 从文件末尾开始搜索
            if model_save_string in line:
                path = line.split('[')[1].split(']')[0]
                return path
    
def extract_best_mae_model_path_from_log_folder(log_folder_path):
    """
    从日志文件夹中提取best_mae_model的保存路径
    
    参数:
        log_folder_path: 日志文件夹的路径
    
    返回:
        best_mae_model的路径，如果未找到则返回None
    """
    # 遍历日志文件夹中的所有文件
    testset_dict = {}
    for file in os.listdir(log_folder_path):
        if file.endswith('.out'):
            log_file_path = os.path.join(log_folder_path, file)
            test_user_list_idx = file.split('_')[-1].split('.')[0].split('testset')[-1]
            model_path = extract_best_mae_model_path(log_file_path)
            if model_path:
                testset_dict[int(test_user_list_idx)] = model_path

    testset_dict = dict(sorted(testset_dict.items()))
    return testset_dict

MASK_PRETRAIN_MODEL_TYPE = 'best_mae_model.pth'
HOMEPATH_ALL_DATA = './checkpoints_Ours_Mask_Pretrain_0d5_/home/mmWave_group/EasyCog/DL_pipeline/configs/Ours/Mask_Pretrain/cfg_CBraMod_Pretrained_asreog_filter3_Mask_Autoencoder_Pretrain.yml/'
MASK_PRETRAIN_MODEL_PATH_DICT_ALL_DATA = {
    0: os.path.join(HOMEPATH_ALL_DATA, '2025_04_27_14_28_47', MASK_PRETRAIN_MODEL_TYPE),
    1: os.path.join(HOMEPATH_ALL_DATA, '2025_04_27_14_28_48', MASK_PRETRAIN_MODEL_TYPE),
    2: os.path.join(HOMEPATH_ALL_DATA, '2025_04_27_14_28_50', MASK_PRETRAIN_MODEL_TYPE),
    3: os.path.join(HOMEPATH_ALL_DATA, '2025_04_27_14_28_54', MASK_PRETRAIN_MODEL_TYPE),
    4: os.path.join(HOMEPATH_ALL_DATA, '2025_04_27_14_28_55', MASK_PRETRAIN_MODEL_TYPE),
    5: os.path.join(HOMEPATH_ALL_DATA, '2025_04_27_14_28_57', MASK_PRETRAIN_MODEL_TYPE),
    6: os.path.join(HOMEPATH_ALL_DATA, '2025_04_27_14_29_01', MASK_PRETRAIN_MODEL_TYPE),
    7: os.path.join(HOMEPATH_ALL_DATA, '2025_04_27_14_29_02', MASK_PRETRAIN_MODEL_TYPE),
    8: os.path.join(HOMEPATH_ALL_DATA, '2025_04_27_14_29_05', MASK_PRETRAIN_MODEL_TYPE),
    9: os.path.join(HOMEPATH_ALL_DATA, '2025_04_27_14_29_08', MASK_PRETRAIN_MODEL_TYPE),
}

HOMEPATH_CLEAN_DATA = './checkpoints_Ours_Mask_Pretrain_0d5_/home/mmWave_group/EasyCog/DL_pipeline/configs/Ours/Mask_Pretrain/cfg_CBraMod_Pretrained_asreog_filter3_Mask_Autoencoder_Pretrain.yml/'
MASK_PRETRAIN_MODEL_PATH_DICT_CLEAN_DATA = {
    0: os.path.join(HOMEPATH_CLEAN_DATA, '2025_04_27_15_22_15', MASK_PRETRAIN_MODEL_TYPE),
    1: os.path.join(HOMEPATH_CLEAN_DATA, '2025_04_27_15_22_19', MASK_PRETRAIN_MODEL_TYPE),
    2: os.path.join(HOMEPATH_CLEAN_DATA, '2025_04_27_15_22_20', MASK_PRETRAIN_MODEL_TYPE),
    3: os.path.join(HOMEPATH_CLEAN_DATA, '2025_04_27_15_22_22', MASK_PRETRAIN_MODEL_TYPE),
    4: os.path.join(HOMEPATH_CLEAN_DATA, '2025_04_27_15_22_26', MASK_PRETRAIN_MODEL_TYPE),
    5: os.path.join(HOMEPATH_CLEAN_DATA, '2025_04_27_15_22_29', MASK_PRETRAIN_MODEL_TYPE),
    6: os.path.join(HOMEPATH_CLEAN_DATA, '2025_04_27_15_22_31', MASK_PRETRAIN_MODEL_TYPE),
    7: os.path.join(HOMEPATH_CLEAN_DATA, '2025_04_27_15_22_35', MASK_PRETRAIN_MODEL_TYPE),
    8: os.path.join(HOMEPATH_CLEAN_DATA, '2025_04_27_15_22_37', MASK_PRETRAIN_MODEL_TYPE),
    9: os.path.join(HOMEPATH_CLEAN_DATA, '2025_04_27_15_22_39', MASK_PRETRAIN_MODEL_TYPE),
}

# 使用示例
if __name__ == "__main__":

    print("all_0426")
    testset_dict = extract_best_mae_model_path_from_log_folder(
        "/home/mmWave_group/EasyCog/logs/Ours/Mask_Pretrain/all_0426/")
    # print(testset_dict)
    print("{")
    for testset_idx, model_path in testset_dict.items():
        print(f"{testset_idx}: '{model_path}',")
    print("}")


    print("clean_0426")
    testset_dict = extract_best_mae_model_path_from_log_folder(
        "/home/mmWave_group/EasyCog/logs/Ours/Mask_Pretrain/clean_0426/")
    # print(testset_dict)
    print("{")
    for testset_idx, model_path in testset_dict.items():
        print(f"{testset_idx}: '{model_path}',")
    print("}")