import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random
import itertools
from matplotlib.patches import Circle, Polygon
from scipy.interpolate import griddata
from scipy.signal import butter, sosfiltfilt
import pywt

BIAS_AUG_WEIGHT = [14.14, 11.54, 6.32, 6.32, 6.32, 8.97]        # 1/sqrt(n)

MOCA_TASK_SCORE_MAX = [5, 3, 6, 3, 2, 5, 6] # visuo-spatial, naming, attention, language, abstract, memory, orientation
VIDEO_TASK_TO_MOCA_TASK_MAPPING = [
    [0, 2],     # naturalistic pictures:            visuo-spatial and attention
    [0, 2],     # complex pictures:                 visuo-spatial and attention
    [0, 4],     # social and non-social situation:  visuo-spatial and abstract
    [0, 1, 4],  # incongruous:                   visuo-spatial, naming and abstract
    [2, 4],     # social scene:                     attention and abstract
    [0, 2],     # Moving targets:                   visuo-spatial and attention
    [3, 6],     # incongruent text:                 language and orientation
    [2, 5],     # memory:                           attention and memory
    [2, 6],     # calculation:                      attention and orientation
    [0,1,2,3,4,5,6]     # resting
]

VIDEO_TASK_MAPPING_MOCA_MATRIX = np.zeros((7,10))
for i in range(7):
    for j in range(9):
        # from i to j
        if i in VIDEO_TASK_TO_MOCA_TASK_MAPPING[j]:
            VIDEO_TASK_MAPPING_MOCA_MATRIX[i,j] = 1/len(VIDEO_TASK_TO_MOCA_TASK_MAPPING[j])/MOCA_TASK_SCORE_MAX[i]
for i in range(7):
    VIDEO_TASK_MAPPING_MOCA_MATRIX[i,9] = 1/30
PINV_VIDEO_TASK_MAPPING_MOCA_MATRIX = np.linalg.pinv(VIDEO_TASK_MAPPING_MOCA_MATRIX)


MMSE_TASK_SCORE_MAX = [10, 6, 5, 2, 6, 1]
VIDEO_TASK_TO_MMSE_TASK_MAPPING = [
    [2,5], # naturalistic pictures:            attention and visuo-spatial
    [2,5], # complex pictures:                 attention and visuo-spatial
    [3,5], # social and non-social situation:  naming and visuo-spatial
    [0,3], # incongruous:                      Orientation and naming
    [2,3], # social scene:                     attention and naming
    [2,5], # Moving targets:                   attention and visuo-spatial
    [0,4], # incongruent text:                 orientation and language
    [1,2], # memory:                           memory and attention
    [0,2], # calculation:                      attention and orientation
]

VIDEO_TASK_MAPPING_MMSE_MATRIX = np.zeros((6,10))
for i in range(6):
    for j in range(9):
        # from i to j
        if i in VIDEO_TASK_TO_MMSE_TASK_MAPPING[j]:
            VIDEO_TASK_MAPPING_MMSE_MATRIX[i,j] = 1/len(VIDEO_TASK_TO_MMSE_TASK_MAPPING[j])/MMSE_TASK_SCORE_MAX[i]

for i in range(6):
    VIDEO_TASK_MAPPING_MMSE_MATRIX[i,9] = 1/30
PINV_VIDEO_TASK_MAPPING_MMSE_MATRIX = np.linalg.pinv(VIDEO_TASK_MAPPING_MMSE_MATRIX)



# FR1~FR4 is the frontal channels (from right to left 1-4), R1~R10 is the right channels, L1~L10 is the left channels

### Channel setting for all formal data
CH_NAMES = ['L1', 'L2', 'L3', 'FR4', 'FR3', 'L8', 'L9', 'L10', 'R1', 'R2', 'R3', 'FR2', 'FR1', 'R8', 'R9', 'R10']   # 头带出线口在下侧/脸右侧

STD_CH = ['R1', 'R2', 'R3', 'FR1', 'FR2', 'R8', 'R9', 'R10', 'L1', 'L2', 'L3', 'FR4', 'FR3', 'L8', 'L9', 'L10']     # 标准的通道顺序
STD_CH_DIST = [[0,0], [-1,1], [-2,0], [3,2], [4,2], [-2,-2], [-1,-3], [0,-2], [9,0], [10,1], [11,0], [6,2], [5,2], [11,-2], [10,-3], [9,-2]]

FPS = 30
TASK_DURATION = round((430*FPS+2)/FPS)
TASK_1_START = round((4*FPS+2)/FPS)
TASK_1_PIC_START = [4*FPS+2, 9*FPS, 14*FPS, 19*FPS, 24*FPS, 29*FPS, 34*FPS, 39*FPS, 44*FPS, 49*FPS]
TASK_1_PIC_START = [round(i/FPS) for i in TASK_1_PIC_START]
TASK_1_END = round((54*FPS)/FPS)
TASK_2_START = round((56*FPS)/FPS)
TASK_2_PIC_START = [56*FPS, 61*FPS, 66*FPS, 71*FPS, 76*FPS, 81*FPS, 86*FPS, 91*FPS, 96*FPS, 101*FPS]
TASK_2_PIC_START = [round(i/FPS) for i in TASK_2_PIC_START]
TASK_2_END = round((106*FPS)/FPS)
TASK_3_START = round((107*FPS+29)/FPS)
TASK_3_PIC_START = [107*FPS+29, 112*FPS+29, 117*FPS+29, 122*FPS+29, 127*FPS+29, 132*FPS+29, 137*FPS+29, 142*FPS+29, 147*FPS+29, 152*FPS+29]
TASK_3_PIC_START = [round(i/FPS) for i in TASK_3_PIC_START]
TASK_3_END = round((157*FPS+29)/FPS)
TASK_4_START = round((159*FPS+29)/FPS)
TASK_4_PIC_START = [159*FPS+29, 164*FPS+29, 169*FPS+29, 174*FPS+29, 179*FPS+29, 184*FPS+29, 189*FPS+29, 194*FPS+29, 199*FPS+29, 204*FPS+29]
TASK_4_PIC_START = [round(i/FPS) for i in TASK_4_PIC_START]
TASK_4_END = round((209*FPS+29)/FPS)
TASK_5_START = round((211*FPS+29)/FPS)
TASK_5_PIC_START = [211*FPS+29, 216*FPS+29, 221*FPS+29, 226*FPS+29, 231*FPS+29, 236*FPS+29, 241*FPS+29, 246*FPS+29, 251*FPS+29, 256*FPS+29]
TASK_5_PIC_START = [round(i/FPS) for i in TASK_5_PIC_START]
TASK_5_END = round((261*FPS+29)/FPS)
TASK_6_START = round((263*FPS+29)/FPS)
TASK_6_PIC_START = [263*FPS+29, 268*FPS+29, 273*FPS+29, 279*FPS+2, 284*FPS+2, 289*FPS+2]
TASK_6_PIC_START = [round(i/FPS) for i in TASK_6_PIC_START]
TASK_6_END = round((294*FPS+2)/FPS)
TASK_7_START = round((296*FPS+2)/FPS)
TASK_7_PIC_START = [296*FPS+2, 301*FPS+2, 306*FPS+2, 311*FPS+2, 316*FPS+2, 321*FPS+2, 326*FPS+2, 331*FPS+2, 336*FPS+2, 341*FPS+2]
TASK_7_PIC_START = [round(i/FPS) for i in TASK_7_PIC_START]
TASK_7_END = round((346*FPS+2)/FPS)
TASK_8_START = round((348*FPS+2)/FPS)
TASK_8_PIC_START = [348*FPS+2, 353*FPS+2, 358*FPS+2, 363*FPS+2, 368*FPS+2, 373*FPS+2, 378*FPS+2, 383*FPS+2, 388*FPS+2, 393*FPS+2]
TASK_8_PIC_START = [round(i/FPS) for i in TASK_8_PIC_START]
TASK_8_END = round((398*FPS+2)/FPS)
TASK_9_START = round((400*FPS+2)/FPS)
TASK_9_PIC_START = [400*FPS+2, 403*FPS+2, 406*FPS+2, 409*FPS+2, 412*FPS+2, 415*FPS+2, 418*FPS+2, 421*FPS+2, 424*FPS+2, 427*FPS+2]
TASK_9_PIC_START = [round(i/FPS) for i in TASK_9_PIC_START]
TASK_9_END = round((430*FPS+2)/FPS)

TASK_SLOT = [TASK_1_START, TASK_2_START, TASK_3_START, TASK_4_START, TASK_5_START, TASK_6_START, TASK_7_START, TASK_8_START, TASK_9_START]
TASK_PIC_SLOT = [TASK_1_PIC_START, TASK_2_PIC_START, TASK_3_PIC_START, TASK_4_PIC_START, TASK_5_PIC_START, TASK_6_PIC_START, TASK_7_PIC_START, TASK_8_PIC_START, TASK_9_PIC_START]

RESTING_DURATION = 3*60

def get_biased_aug_times(moca_score, n=1):
    assert moca_score <= 1
    if moca_score <= 5/30:
        aug_times_weight = BIAS_AUG_WEIGHT[0] / n
    elif moca_score<= 10/30:
        aug_times_weight = BIAS_AUG_WEIGHT[1] / n
    elif moca_score <= 15/30:
        aug_times_weight = BIAS_AUG_WEIGHT[2] / n
    elif moca_score <= 20/30:
        aug_times_weight = BIAS_AUG_WEIGHT[3] / n
    elif moca_score <= 25/30:
        aug_times_weight = BIAS_AUG_WEIGHT[4] / n
    else:
        aug_times_weight = BIAS_AUG_WEIGHT[5] / n
    return aug_times_weight

def read_data(path, type):
    data = np.load(os.path.join(path, 'exg_data.npz'))
    exg_data = data['data'][data['exg_channels'],:]
    exg_timestamp = data['data'][data['timestamp_channel'],:]
    sampling_rate = float(data['sampling_rate'])

    camera_timestamps_webcam = np.load(os.path.join(path, 'recorded_video_frame_timestamps_in_record_webcam_function.npy'))
    camera_timestamps = np.loadtxt(os.path.join(path, 'recorded_video_frame_timestamps.txt'))

    start_points = np.load(os.path.join(path, 'timestamps.npz'))

    if type=='video':
        et_ts, et_data = read_eye_tracking_txt(os.path.join(path, 'eye_tracking_record.txt'))
    elif type == 'resting':
        et_ts, et_data = None, None
    return exg_data, exg_timestamp, sampling_rate, start_points, camera_timestamps_webcam, camera_timestamps, et_ts, et_data

def read_data_debug(path):
    data = np.load(os.path.join(path, 'exg_data.npz'))
    exg_data = data['data'][data['exg_channels'],:]
    exg_timestamp = data['data'][data['timestamp_channel'],:]
    sampling_rate = float(data['sampling_rate'])

    camera_timestamps_webcam = np.load(os.path.join(path, 'recorded_video_frame_timestamps_in_record_webcam_function.npy'))
    camera_timestamps = np.loadtxt(os.path.join(path, 'recorded_video_frame_timestamps.txt'))

    return exg_data, exg_timestamp, sampling_rate, camera_timestamps_webcam, camera_timestamps

def read_eye_tracking_txt(path):
    with open(path, 'r') as file:
        data = file.readlines()
    
    for i in range(len(data)-1, -1, -1):
        if "TobiiStream" not in data[i]:
            data.pop(i)

    # Split the data into a list of strings
    data = [line.split(' ') for line in data]
    ts = [line[0] for line in data]
    data = [line[-2:] for line in data]


    # for i in range(len(ts)):
    #     print(ts[i])
    #     print(data[i])


    # Convert the strings to floats
    ts = np.array(ts, dtype=float)
    data = np.array(data, dtype=float)

    return ts, data

def smooth_data(data, window_size=3):
    """
    Smooth the data using a moving average filter.
    
    Parameters
    ----------
    data : numpy.ndarray
        The data to be smoothed.
    window_size : int
        The size of the moving average window.

    
    Returns
    -------
    smoothed_data : numpy.ndarray
        The smoothed data.
    """
    # Create the moving average filter
    filter = np.ones(window_size) / window_size
    
    # Apply the filter to the data
    smoothed_data = np.convolve(data, filter, mode='same')
    
    return smoothed_data

def split_resting_exg(exg_data, sps):
    return exg_data

def split_tasks_exg(exg_data, eye_tracking_data, sps, video_duration, timestamp=None):
    real_fps_scale = TASK_DURATION/video_duration
    sps = round(sps / real_fps_scale)  # if video play faster, then the relative sps will be smaller to the video
    """
    E.g., normally when video play 30 frames, 1s pass by and sps samples are collected, however
    if video play 60 frames per second (faster than normally), than only sps/2 samples are collected
    """

    # if exg_data.shape[1] < TASK_DURATION*sps:
    #     return None
    
    task_data = {}
    task_et = {}
    task_data['task1'] = exg_data[:, round(TASK_1_START*sps):round(TASK_1_END*sps)]
    task_data['task2'] = exg_data[:, round(TASK_2_START*sps):round(TASK_2_END*sps)]
    task_data['task3'] = exg_data[:, round(TASK_3_START*sps):round(TASK_3_END*sps)]
    task_data['task4'] = exg_data[:, round(TASK_4_START*sps):round(TASK_4_END*sps)]
    task_data['task5'] = exg_data[:, round(TASK_5_START*sps):round(TASK_5_END*sps)]
    task_data['task6'] = exg_data[:, round(TASK_6_START*sps):round(TASK_6_END*sps)]
    task_data['task7'] = exg_data[:, round(TASK_7_START*sps):round(TASK_7_END*sps)]
    task_data['task8'] = exg_data[:, round(TASK_8_START*sps):round(TASK_8_END*sps)]
    task_data['task9'] = exg_data[:, round(TASK_9_START*sps):round(TASK_9_END*sps)]

    task_et['task1'] = eye_tracking_data[round(TASK_1_START*sps):round(TASK_1_END*sps)]
    task_et['task2'] = eye_tracking_data[round(TASK_2_START*sps):round(TASK_2_END*sps)]
    task_et['task3'] = eye_tracking_data[round(TASK_3_START*sps):round(TASK_3_END*sps)]
    task_et['task4'] = eye_tracking_data[round(TASK_4_START*sps):round(TASK_4_END*sps)]
    task_et['task5'] = eye_tracking_data[round(TASK_5_START*sps):round(TASK_5_END*sps)]
    task_et['task6'] = eye_tracking_data[round(TASK_6_START*sps):round(TASK_6_END*sps)]
    task_et['task7'] = eye_tracking_data[round(TASK_7_START*sps):round(TASK_7_END*sps)]
    task_et['task8'] = eye_tracking_data[round(TASK_8_START*sps):round(TASK_8_END*sps)]
    task_et['task9'] = eye_tracking_data[round(TASK_9_START*sps):round(TASK_9_END*sps)]

    task_data['task1_pic'] = []
    for i in range(len(TASK_1_PIC_START)-1):
        task_data['task1_pic'].append(exg_data[:, round(TASK_1_PIC_START[i]*sps):round(TASK_1_PIC_START[i+1]*sps)])
    task_data['task1_pic'].append(exg_data[:, round(TASK_1_PIC_START[-1]*sps):round(TASK_1_END*sps)])

    task_data['task2_pic'] = []
    for i in range(len(TASK_2_PIC_START)-1):
        task_data['task2_pic'].append(exg_data[:, round(TASK_2_PIC_START[i]*sps):round(TASK_2_PIC_START[i+1]*sps)])
    task_data['task2_pic'].append(exg_data[:, round(TASK_2_PIC_START[-1]*sps):round(TASK_2_END*sps)])

    task_data['task3_pic'] = []
    for i in range(len(TASK_3_PIC_START)-1):
        task_data['task3_pic'].append(exg_data[:, round(TASK_3_PIC_START[i]*sps):round(TASK_3_PIC_START[i+1]*sps)])
    task_data['task3_pic'].append(exg_data[:, round(TASK_3_PIC_START[-1]*sps):round(TASK_3_END*sps)])

    task_data['task4_pic'] = []
    for i in range(len(TASK_4_PIC_START)-1):
        task_data['task4_pic'].append(exg_data[:, round(TASK_4_PIC_START[i]*sps):round(TASK_4_PIC_START[i+1]*sps)])
    task_data['task4_pic'].append(exg_data[:, round(TASK_4_PIC_START[-1]*sps):round(TASK_4_END*sps)])

    task_data['task5_pic'] = []
    for i in range(len(TASK_5_PIC_START)-1):
        task_data['task5_pic'].append(exg_data[:, round(TASK_5_PIC_START[i]*sps):round(TASK_5_PIC_START[i+1]*sps)])
    task_data['task5_pic'].append(exg_data[:, round(TASK_5_PIC_START[-1]*sps):round(TASK_5_END*sps)])

    task_data['task6_pic'] = []
    for i in range(len(TASK_6_PIC_START)-1):
        task_data['task6_pic'].append(exg_data[:, round(TASK_6_PIC_START[i]*sps):round(TASK_6_PIC_START[i+1]*sps)])
    task_data['task6_pic'].append(exg_data[:, round(TASK_6_PIC_START[-1]*sps):round(TASK_6_END*sps)])

    task_data['task7_pic'] = []
    for i in range(len(TASK_7_PIC_START)-1):
        task_data['task7_pic'].append(exg_data[:, round(TASK_7_PIC_START[i]*sps):round(TASK_7_PIC_START[i+1]*sps)])
    task_data['task7_pic'].append(exg_data[:, round(TASK_7_PIC_START[-1]*sps):round(TASK_7_END*sps)])

    task_data['task8_pic'] = []
    for i in range(len(TASK_8_PIC_START)-1):
        task_data['task8_pic'].append(exg_data[:, round(TASK_8_PIC_START[i]*sps):round(TASK_8_PIC_START[i+1]*sps)])
    task_data['task8_pic'].append(exg_data[:, round(TASK_8_PIC_START[-1]*sps):round(TASK_8_END*sps)])

    task_data['task9_pic'] = []
    for i in range(len(TASK_9_PIC_START)-1):
        task_data['task9_pic'].append(exg_data[:, round(TASK_9_PIC_START[i]*sps):round(TASK_9_PIC_START[i+1]*sps)])
    task_data['task9_pic'].append(exg_data[:, round(TASK_9_PIC_START[-1]*sps):round(TASK_9_END*sps)])

    
    task_et['task1_pic'] = []
    for i in range(len(TASK_1_PIC_START)-1):
        task_et['task1_pic'].append(eye_tracking_data[round(TASK_1_PIC_START[i]*sps):round(TASK_1_PIC_START[i+1]*sps)])
    task_et['task1_pic'].append(eye_tracking_data[round(TASK_1_PIC_START[-1]*sps):round(TASK_1_END*sps)])

    task_et['task2_pic'] = []
    for i in range(len(TASK_2_PIC_START)-1):
        task_et['task2_pic'].append(eye_tracking_data[round(TASK_2_PIC_START[i]*sps):round(TASK_2_PIC_START[i+1]*sps)])
    task_et['task2_pic'].append(eye_tracking_data[round(TASK_2_PIC_START[-1]*sps):round(TASK_2_END*sps)])

    task_et['task3_pic'] = []
    for i in range(len(TASK_3_PIC_START)-1):
        task_et['task3_pic'].append(eye_tracking_data[round(TASK_3_PIC_START[i]*sps):round(TASK_3_PIC_START[i+1]*sps)])
    task_et['task3_pic'].append(eye_tracking_data[round(TASK_3_PIC_START[-1]*sps):round(TASK_3_END*sps)])

    task_et['task4_pic'] = []
    for i in range(len(TASK_4_PIC_START)-1):
        task_et['task4_pic'].append(eye_tracking_data[round(TASK_4_PIC_START[i]*sps):round(TASK_4_PIC_START[i+1]*sps)])
    task_et['task4_pic'].append(eye_tracking_data[round(TASK_4_PIC_START[-1]*sps):round(TASK_4_END*sps)])

    task_et['task5_pic'] = []
    for i in range(len(TASK_5_PIC_START)-1):
        task_et['task5_pic'].append(eye_tracking_data[round(TASK_5_PIC_START[i]*sps):round(TASK_5_PIC_START[i+1]*sps)])
    task_et['task5_pic'].append(eye_tracking_data[round(TASK_5_PIC_START[-1]*sps):round(TASK_5_END*sps)])

    task_et['task6_pic'] = []
    for i in range(len(TASK_6_PIC_START)-1):
        task_et['task6_pic'].append(eye_tracking_data[round(TASK_6_PIC_START[i]*sps):round(TASK_6_PIC_START[i+1]*sps)])
    task_et['task6_pic'].append(eye_tracking_data[round(TASK_6_PIC_START[-1]*sps):round(TASK_6_END*sps)])

    task_et['task7_pic'] = []
    for i in range(len(TASK_7_PIC_START)-1):
        task_et['task7_pic'].append(eye_tracking_data[round(TASK_7_PIC_START[i]*sps):round(TASK_7_PIC_START[i+1]*sps)])
    task_et['task7_pic'].append(eye_tracking_data[round(TASK_7_PIC_START[-1]*sps):round(TASK_7_END*sps)])

    task_et['task8_pic'] = []
    for i in range(len(TASK_8_PIC_START)-1):
        task_et['task8_pic'].append(eye_tracking_data[round(TASK_8_PIC_START[i]*sps):round(TASK_8_PIC_START[i+1]*sps)])
    task_et['task8_pic'].append(eye_tracking_data[round(TASK_8_PIC_START[-1]*sps):round(TASK_8_END*sps)])

    task_et['task9_pic'] = []
    for i in range(len(TASK_9_PIC_START)-1):
        task_et['task9_pic'].append(eye_tracking_data[round(TASK_9_PIC_START[i]*sps):round(TASK_9_PIC_START[i+1]*sps)])
    task_et['task9_pic'].append(eye_tracking_data[round(TASK_9_PIC_START[-1]*sps):round(TASK_9_END*sps)])

    if timestamp is not None:
        task_data['task1_timestamp'] = timestamp[round(TASK_1_START*sps):round(TASK_1_END*sps)]
        task_data['task2_timestamp'] = timestamp[round(TASK_2_START*sps):round(TASK_2_END*sps)]
        task_data['task3_timestamp'] = timestamp[round(TASK_3_START*sps):round(TASK_3_END*sps)]
        task_data['task4_timestamp'] = timestamp[round(TASK_4_START*sps):round(TASK_4_END*sps)]
        task_data['task5_timestamp'] = timestamp[round(TASK_5_START*sps):round(TASK_5_END*sps)]
        task_data['task6_timestamp'] = timestamp[round(TASK_6_START*sps):round(TASK_6_END*sps)]
        task_data['task7_timestamp'] = timestamp[round(TASK_7_START*sps):round(TASK_7_END*sps)]
        task_data['task8_timestamp'] = timestamp[round(TASK_8_START*sps):round(TASK_8_END*sps)]
        task_data['task9_timestamp'] = timestamp[round(TASK_9_START*sps):round(TASK_9_END*sps)]

    return task_data, task_et

def channel_remapping(data):
    """
    data: channel*sample
    Weight input order: 
    [Right ear 1, right ear 2, right ear 3, Af8 (side forehead at right), 
    Fp2 (near to center forehead at right), right ear 6, right ear 7, right ear 8,
    left ear 1, left ear 2, left ear 3, Af7 (side forehead at left),
    Fp1 (near to center forehead at left), left ear 6, left ear 7, left ear 8]
    """
    mapping = [8, 9, 10, 12, 11, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]
    data = np.array([data[i,:] for i in mapping])
    return data


def has_common_element(arr1, arr2):
    return bool(set(arr1) & set(arr2))

def get_blink_channel(data_std_ch, bad_chs=None):
    """
    data_std_ch: 16*sample with standard channel order 
    """
    # Blinking channel

    forehead_chs = [3, 4, 11, 12]
    L1_L2 = [8, 9]
    R1_R2 = [0, 1]
    if has_common_element(forehead_chs, bad_chs) is False:
        blink_ch = np.sum(data_std_ch[3:5]+data_std_ch[11:13], axis=0)/4    # 4 forehead channels 
    
    elif has_common_element((R1_R2+L1_L2), bad_chs) is False:
        # no forehead channels
        blink_ch = np.sum(data_std_ch[0:2]+data_std_ch[8:10], axis=0)/4        # R1, R2, L1, L2
    
    elif has_common_element(R1_R2, bad_chs) is False:
        blink_ch = np.sum(data_std_ch[8:10], axis=0)/2    # R1, R2
    
    elif has_common_element(L1_L2, bad_chs) is False:
        blink_ch = np.sum(data_std_ch[0:2], axis=0)/2    # L1, L2
    
    else:
        raise NotImplementedError("Not supported.")

    return blink_ch

def get_eog_channel(data_std_ch, bad_chs=None):
    """
    data_std_ch: 16*sample with standard channel order 
    """
    bad_chs = []
    for i in range(data_std_ch.shape[0]):
        exg_std = data_std_ch[i,:].std()
        # print(i, valid_exg[i, 10*int(sps):].std())
        if  exg_std < 0.1:
            bad_chs.append(i)

    R1_L1 = [0, 8]
    R2_L2 = [1, 9]
    FR1_FR4 = [3, 11]
    if has_common_element(FR1_FR4, bad_chs) is False:
        # EOG channel is the difference between the left and right ear 
        eog_ch_2 = data_std_ch[3, :] - data_std_ch[11, :] # FR1 - FR4
        eog_ch = eog_ch_2
    elif has_common_element(R1_L1, bad_chs) is False:
        eog_ch = data_std_ch[0,:] - data_std_ch[8,:]
    elif has_common_element(R2_L2, bad_chs) is False:
        eog_ch = data_std_ch[1,:] - data_std_ch[9,:]
    else:
        eog_ch = data_std_ch[3, :] - data_std_ch[11, :]
    
    R1_R10 = [0, 7]
    R2_R9 = [1, 6]
    L1_L10 = [8, 15]
    L2_L9 = [9, 14]
    cnt = 0
    if has_common_element(R1_R10, bad_chs) is False:
        eog_ch_y1 = data_std_ch[0,:] - data_std_ch[7,:]
        cnt+=1
    else:
        eog_ch_y1 = 0

    if has_common_element(R2_R9, bad_chs) is False:
        eog_ch_y2 = data_std_ch[1,:] - data_std_ch[6,:]
        cnt+=1
    else:
        eog_ch_y2 = 0

    if has_common_element(L1_L10, bad_chs) is False:
        eog_ch_y3 = data_std_ch[8,:] - data_std_ch[15,:]
        cnt+=1
    else:
        eog_ch_y3 = 0

    if has_common_element(L2_L9, bad_chs) is False:
        eog_ch_y4 = data_std_ch[9,:] - data_std_ch[14,:]
        cnt+=1
    else:
        eog_ch_y4 = 0

    if cnt == 0:
        raise NotImplementedError("Not supported.")

    eog_ch_y = (eog_ch_y1 + eog_ch_y2 + eog_ch_y3 + eog_ch_y4)/cnt

    return np.vstack([eog_ch, eog_ch_y])

def channel_norm(data):
    """
    data: one channel*sample

    Normalize the data to mwan 0 and std 1
    """
    data = (data - np.mean(data))/np.std(data)
    return data

def check_completence(path, type):
    """
    Check whether the given dir contains all valid files
    """
    if not os.path.exists(os.path.join(path, 'exg_data.npz')):
        print("exg_data.npz not found")
        return False
    if type=='video':
        if not os.path.exists(os.path.join(path, 'eye_tracking_record.txt')):
            print("eye_tracking_record.txt not found")
            return False
    if not os.path.exists(os.path.join(path, 'recorded_video_frame_timestamps_in_record_webcam_function.npy')):
        print("recorded_video_frame_timestamps_in_record_webcam_function.npy not found")
        return False
    if not os.path.exists(os.path.join(path, 'recorded_video_frame_timestamps.txt')):
        print("recorded_video_frame_timestamps.txt not found")
        return False
    if not os.path.exists(os.path.join(path, 'timestamps.npz')):
        print("timestamps.npz not found")
        return False
    # if not os.path.exists(os.path.join(path, 'video_task.avi')) and not os.path.exists(os.path.join(path, 'resting_task.avi')):
    #     print("video avi file not found")
    #     return False
    return True
