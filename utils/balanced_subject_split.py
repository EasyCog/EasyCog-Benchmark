import numpy as np

class Score_Balanced_Subjects_All:
    #### This group has the clean and noisy data
    """
    Fold 0: 用户数 = 7, 平均MoCA = 17.29, 平均MMSE = 22.57
    Fold 1: 用户数 = 7, 平均MoCA = 17.29, 平均MMSE = 22.57
    Fold 2: 用户数 = 7, 平均MoCA = 17.43, 平均MMSE = 22.57
    Fold 3: 用户数 = 7, 平均MoCA = 17.29, 平均MMSE = 22.71
    Fold 4: 用户数 = 7, 平均MoCA = 17.29, 平均MMSE = 22.71
    Fold 5: 用户数 = 7, 平均MoCA = 17.29, 平均MMSE = 22.57
    Fold 6: 用户数 = 7, 平均MoCA = 17.29, 平均MMSE = 22.57
    Fold 7: 用户数 = 7, 平均MoCA = 17.14, 平均MMSE = 22.57
    Fold 8: 用户数 = 7, 平均MoCA = 17.29, 平均MMSE = 22.57
    Fold 9: 用户数 = 6, 平均MoCA = 17.33, 平均MMSE = 22.67
    """
    test_users_0 = ['017_patient', '030_patient', '031_patient', '033_patient', '038_patient', '047_patient', '061_patient']
    test_users_1 = ['026_patient', '028_patient', '029_patient', '042_patient', '051_patient', '069_patient', '072_patient']
    test_users_2 = ['006_patient', '014_patient', '023_patient', '037_patient', '043_patient', '055_patient', '071_patient']
    test_users_3 = ['011_patient', '025_patient', '034_patient', '036_patient', '050_patient', '058_patient', '074_patient']
    test_users_4 = ['005_patient', '012_patient', '015_patient', '035_patient', '041_patient', '059_patient', '065_patient']
    test_users_5 = ['009_patient', '024_patient', '027_patient', '049_patient', '057_patient', '067_patient', '073_patient']
    test_users_6 = ['002_patient', '018_patient', '052_patient', '062_patient', '066_patient', '070_patient', '075_patient']
    test_users_7 = ['013_patient', '022_patient', '032_patient', '046_patient', '054_patient', '064_patient', '068_patient']
    test_users_8 = ['007_patient', '016_patient', '019_patient', '020_patient', '039_patient', '045_patient', '060_patient']
    test_users_9 = ['003_patient', '004_patient', '008_patient', '040_patient', '048_patient', '056_patient']
    total_users_lists = [test_users_0, test_users_1, test_users_2, test_users_3, test_users_4, test_users_5, test_users_6, test_users_7, test_users_8, test_users_9]



class Score_Balanced_Subjects_Clean:
    ### This subject group has a cleaner data
    """
    Fold 0: 用户数 = 5, 平均MoCA = 17.80, 平均MMSE = 23.00
    Fold 1: 用户数 = 5, 平均MoCA = 18.00, 平均MMSE = 23.00
    Fold 2: 用户数 = 5, 平均MoCA = 17.80, 平均MMSE = 23.00
    Fold 3: 用户数 = 5, 平均MoCA = 17.80, 平均MMSE = 23.00
    Fold 4: 用户数 = 5, 平均MoCA = 18.00, 平均MMSE = 23.20
    Fold 5: 用户数 = 5, 平均MoCA = 17.80, 平均MMSE = 23.00
    Fold 6: 用户数 = 5, 平均MoCA = 17.80, 平均MMSE = 23.00
    Fold 7: 用户数 = 5, 平均MoCA = 18.00, 平均MMSE = 23.00
    Fold 8: 用户数 = 5, 平均MoCA = 17.80, 平均MMSE = 23.00
    Fold 9: 用户数 = 4, 平均MoCA = 17.75, 平均MMSE = 23.00
    """
    test_users_0 = ['002_patient', '030_patient', '045_patient', '057_patient', '072_patient']
    test_users_1 = ['005_patient', '016_patient', '018_patient', '020_patient', '068_patient']
    test_users_2 = ['011_patient', '032_patient', '047_patient', '048_patient', '054_patient']
    test_users_3 = ['006_patient', '023_patient', '027_patient', '036_patient', '073_patient']
    test_users_4 = ['043_patient', '050_patient', '051_patient', '066_patient', '069_patient']
    test_users_5 = ['013_patient', '015_patient', '017_patient', '065_patient', '071_patient']
    test_users_6 = ['008_patient', '026_patient', '038_patient', '039_patient', '058_patient']
    test_users_7 = ['007_patient', '014_patient', '024_patient', '034_patient', '059_patient']
    test_users_8 = ['028_patient', '049_patient', '060_patient', '067_patient', '070_patient']
    test_users_9 = ['033_patient', '037_patient', '041_patient', '074_patient']
    total_users_lists = [test_users_0, test_users_1, test_users_2, test_users_3, test_users_4, test_users_5, test_users_6, test_users_7, test_users_8, test_users_9]
