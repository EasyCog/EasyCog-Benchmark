import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.abspath(''))) + '/../')
sys.path.append(os.path.dirname(os.path.abspath(os.path.abspath(''))))
sys.path.append('/home/mmWave_group/EasyCog/')
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from data_processing.analysis_utils import TASK_PIC_SLOT, STD_CH
import cv2
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter


############## USER PATH ###############
video_path = f'path/to/video.mp4'
features_folder = 'dataset_path/sliced'
feat_name = 'asreog_filter_order3_all_data'
sliced_path = f'{features_folder}/{feat_name}'
save_path = 'path/to/save/figs'
######################################

screen_resolution = [2560, 1920]


def load_data(sliced_path):

    data_all = {}

    sliced_files = os.listdir(sliced_path)

    cnt = 0
    for file in sliced_files:
        cnt += 1
        print(f"Checking {cnt}/{len(sliced_files)}")
        filepath = os.path.join(sliced_path, file)
        if os.path.isfile(filepath):
            unsliced_data = np.load(filepath, allow_pickle=True)
            subject = str(unsliced_data['subject'])
            data_type = str(unsliced_data['type'])
            date = str(unsliced_data['date'])
            task = file.split('.')[0].split('-')[-4]
            pic = file.split('.')[0].split('-')[-3]
            start = file.split('.')[0].split('-')[-2]
            end = file.split('.')[0].split('-')[-1]
            raw_eeg = unsliced_data['eeg_seg']
            dtf = unsliced_data['DTF']
            if data_type == 'resting':
                eye_tracking = None
            else:
                eye_tracking = unsliced_data['et_seg']
            

            data_all[f'{subject}-{date}-{data_type}-task{task}-pic{pic}-{start}-{end}'] = {'eeg': raw_eeg,
                                                                                            'et': eye_tracking, 
                                                                                            'dtf': dtf, 
                                                                                            }
    return data_all

def get_stft(data, sps=125):
    Pxx_list = []
    for j in range(16): 
        f, t, Pxx = signal.stft(data[j, :], fs=sps, nperseg=256, noverlap=200, nfft=256)
        # normalize the stft
        Pxx = Pxx/np.max(np.abs(Pxx))
        Pxx_list.append(Pxx)
    return f, t, np.array(Pxx_list)

def get_pic_from_video(target_task, target_pic, video_path):
    timeslot = TASK_PIC_SLOT[target_task][target_pic] + 1
    
    # load the video to extract the frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, timeslot*30)
    ret, frame = cap.read()
    b,g,r = cv2.split(frame)
    frame = cv2.merge([r,g,b])
    # frame = cv2.flip(frame, 0)
    frame = cv2.resize(frame, (screen_resolution[0], screen_resolution[1]))
    return frame

def plot_gaze_distribution(et_subject, savepath):
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.hist(et_subject[:,1], bins=100)
    plt.xlim(0, screen_resolution[0])
    plt.xlabel('x')
    plt.subplot(1,2,2)
    plt.hist(et_subject[:,0], bins=100)
    plt.xlim(0, screen_resolution[1])
    plt.xlabel('y')
    plt.savefig(savepath)
    plt.close()
    return

def calculate_saliency_map(et, circle_radius=100):
    # calculate the saliency map
    saliency_map = np.zeros((screen_resolution[1], screen_resolution[0]))
    
    # exclude the all 0 gaze for drifting elimination using boolean indexing
    valid_gaze_mask = ~((et[:, 0] == 0) & (et[:, 1] == 0))
    et = et[valid_gaze_mask]
    
    # if et is empty after filtering, return an empty map or handle appropriately
    if et.shape[0] == 0:
        # Option 1: Return a zero map (or handle as needed)
        # return np.zeros((screen_resolution[1], screen_resolution[0]), dtype=np.uint8) 
        # Option 2: Or perhaps return the empty RGBA map directly if that's preferred
        saliency_map_rgba = ScalarMappable(cmap='hot').to_rgba(saliency_map)[:,:,:3]
        return np.uint8(saliency_map_rgba*255)

    for i in range(et.shape[0]):
        x = int(et[i,0])
        y = int(et[i,1])
        saliency_map[max(0, y-circle_radius):min(screen_resolution[1], y+circle_radius), max(0, x-circle_radius):min(screen_resolution[0], x+circle_radius)] += 1
    
    # saliency_map[saliency_map < 10] = saliency_map[saliency_map < 10] *
    # saliency_map = np.log(saliency_map+1)
    saliency_map = saliency_map/saliency_map.max()
    # smooth the saliency map
    saliency_map = gaussian_filter(saliency_map, sigma=20, radius=100)
    saliency_map = ScalarMappable(cmap='hot').to_rgba(saliency_map)[:,:,:3]
    return np.uint8(saliency_map*255)


def plot_saliency_map(saliency_map, frame, savepath):
    img_sum = cv2.addWeighted(frame, 0.25, saliency_map, 0.75, 0)
    plt.figure(figsize=(10, 5))
    plt.axis('off')
    plt.imshow(img_sum)
    plt.savefig(savepath)
    plt.close()
    return

def plot_stft(stft_list, f, savepath):
    concat_stft = np.concatenate(stft_list, axis=1)
    plt.figure(figsize=(4, 6))
    plt.pcolormesh(f, np.linspace(0, 16, 16*13), np.abs(concat_stft).T, cmap='jet', shading='gouraud')
    for i in range(15):
        plt.hlines(i+1, 0, 30, colors='w', linestyles='-', linewidth=1)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ticks = np.linspace(0.5, 15.5, 16)
    ax.set_yticks(ticks, labels=STD_CH)
    ax.set_xlabel('Frequency [Hz]', loc='center')
    ax.set_ylabel('Channel', loc='center')
    plt.xlim(0, 30)
    plt.savefig(savepath)
    plt.close()

def plot_stft_for_paper_subplot_version(stft_list, f, savepath):
    left_idx = [8,9,10,13,14,15,11,12]
    right_idx = [0,1,2,5,6,7,3,4]

    plt.figure(figsize=(5.5, 5))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # plot left idx
    for i in range(8):
        plt.subplot(2,8,i+1)
        plt.pcolormesh(np.linspace(0, 1, 13), f, np.abs(stft_list[left_idx[i]]), cmap='jet', shading='gouraud')
        plt.xticks([])
        plt.yticks([])
        plt.ylim(0,30)
        plt.title(STD_CH[left_idx[i]], x=0.5, y=1)
        if i == 0:
            plt.ylabel('Frequency [Hz]', loc='center', fontsize=12)
            plt.yticks(np.linspace(0, 30, 7), labels=['0', '5', '10', '15', '20', '25', '30'])
    # plot right idx
    for i in range(8):
        plt.subplot(2,8,i+9)
        plt.pcolormesh(np.linspace(0, 1, 13), f, np.abs(stft_list[right_idx[i]]), cmap='jet', shading='gouraud')
        plt.xticks([])
        plt.yticks([])
        plt.ylim(0,30)
        plt.title(STD_CH[right_idx[i]], x=0.5, y=-0.2)
        if i == 0:
            plt.ylabel('Frequency [Hz]', loc='center', fontsize=12)
            plt.yticks(np.linspace(0, 30, 7), labels=['0', '5', '10', '15', '20', '25', '30'])
    plt.savefig(savepath)
    plt.close()

def plot_dtf(dtf, savepath):
    dtf = dtf.mean(axis=2)
    # set diagonal to 0
    np.fill_diagonal(dtf, 0)
    plt.figure(figsize=(4, 4))
    plt.matshow(dtf, cmap='jet')
    plt.xticks(np.arange(len(STD_CH)), labels=STD_CH)
    plt.yticks(np.arange(len(STD_CH)), labels=STD_CH)
    plt.colorbar()
    # colorbar range
    plt.clim(0, 1.2)
    plt.savefig(savepath)
    plt.close()
    return

def plot_feature_difference_paper_selection(data_all, subject_HC, subject_MCI, subject_Severe, target_task, target_pic, pic_frame, savepath):
    if not isinstance(data_all, dict):
        data_all = data_all.item()
    data_len=625 # 125*5
    signal_subject_HC = np.zeros((16, data_len))
    et_subject_HC = np.zeros((data_len,2))
    dtf_subject_HC = np.zeros((16, 16, 29))
    signal_subject_MCI = np.zeros((16, data_len))
    et_subject_MCI = np.zeros((data_len,2))
    dtf_subject_MCI = np.zeros((16, 16, 29))
    signal_subject_Severe = np.zeros((16, data_len))
    et_subject_Severe = np.zeros((data_len,2))
    dtf_subject_Severe = np.zeros((16, 16, 29))

    for key, val in data_all.items():
        if subject_HC in key:
            if f'task{target_task}' in key and f'pic{target_pic}' in key:
                start = int(key.split('-')[-2])
                end = int(key.split('-')[-1])
                signal_subject_HC[:, start:end] = val['eeg']
                et_subject_HC[ start:end,:] = val['et']
                dtf_subject_HC += val['dtf']
        elif subject_MCI in key:
            if f'task{target_task}' in key and f'pic{target_pic}' in key:
                start = int(key.split('-')[-2])
                end = int(key.split('-')[-1])
                signal_subject_MCI[:, start:end] = val['eeg']
                et_subject_MCI[ start:end,:] = val['et']
                dtf_subject_MCI += val['dtf']
        elif subject_Severe in key:
            if f'task{target_task}' in key and f'pic{target_pic}' in key:
                start = int(key.split('-')[-2])
                end = int(key.split('-')[-1])
                signal_subject_Severe[:, start:end] = val['eeg']
                et_subject_Severe[ start:end,:] = val['et']
                dtf_subject_Severe += val['dtf']
    
    f_HC, t_HC, stft_list_HC = get_stft(signal_subject_HC)  
    f_MCI, t_MCI, stft_list_MCI = get_stft(signal_subject_MCI)
    f_Severe, t_Severe, stft_list_Severe = get_stft(signal_subject_Severe)
    # plot the stft
    plot_stft_for_paper_subplot_version(stft_list_HC, f_HC, os.path.join(savepath, f'task{target_task}_pic{target_pic}_subject_HC_stft.png'))
    plot_stft_for_paper_subplot_version(stft_list_MCI, f_MCI, os.path.join(savepath, f'task{target_task}_pic{target_pic}_subject_MCI_stft.png'))
    plot_stft_for_paper_subplot_version(stft_list_Severe, f_Severe, os.path.join(savepath, f'task{target_task}_pic{target_pic}_subject_Dementia_stft.png'))
                
    # plot the saliency map
    saliency_map_HC = calculate_saliency_map(et_subject_HC)
    saliency_map_MCI = calculate_saliency_map(et_subject_MCI)
    saliency_map_Severe = calculate_saliency_map(et_subject_Severe)
    plot_saliency_map(saliency_map_HC, pic_frame, os.path.join(savepath, f'task{target_task}_pic{target_pic}_subject_HC_saliency.png'))
    plot_saliency_map(saliency_map_MCI, pic_frame, os.path.join(savepath, f'task{target_task}_pic{target_pic}_subject_MCI_saliency.png'))
    plot_saliency_map(saliency_map_Severe, pic_frame, os.path.join(savepath, f'task{target_task}_pic{target_pic}_subject_Dementia_saliency.png'))

    # plot the gaze distribution
    plot_gaze_distribution(et_subject_HC, os.path.join(savepath, f'task{target_task}_pic{target_pic}_subject_HC_gaze.png'))
    plot_gaze_distribution(et_subject_MCI, os.path.join(savepath, f'task{target_task}_pic{target_pic}_subject_MCI_gaze.png'))
    plot_gaze_distribution(et_subject_Severe, os.path.join(savepath, f'task{target_task}_pic{target_pic}_subject_Dementia_gaze.png'))

    # plot the dtf
    plot_dtf(dtf_subject_HC, os.path.join(savepath, f'task{target_task}_pic{target_pic}_subject_HC_dtf.png'))
    plot_dtf(dtf_subject_MCI, os.path.join(savepath, f'task{target_task}_pic{target_pic}_subject_MCI_dtf.png'))
    plot_dtf(dtf_subject_Severe, os.path.join(savepath, f'task{target_task}_pic{target_pic}_subject_Dementia_dtf.png'))

    plt.close('all')
    return



def main_select_pic(list_target, sliced_path, video_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    data_info_path = os.path.join('/data/mmWave_group/EasyCog/data_check', 'data_info.npz')
    if os.path.exists(data_info_path):
        print(f'Loading data from Existing file: {data_info_path}')
        data_info = np.load(data_info_path, allow_pickle=True)
        data_all = data_info['data_all']
    else:
        print(f'Loading data from New file: {data_info_path}')
        data_all = load_data(sliced_path)
        np.savez(data_info_path, data_all=data_all)
    
    for itm in list_target:
        subject_HC = itm[0]
        subject_MCI = itm[1]
        subject_Severe = itm[2]
        task = itm[3]
        pic = itm[4]
        print(f'Processing subject {subject_HC}, {subject_MCI}, {subject_Severe}, task {task}, pic {pic}')
        pic_frame = get_pic_from_video(task, pic, video_path)
        # plt.figure(figsize=(7, 5))
        # plt.axis('off')
        # plt.imshow(pic_frame)
        # plt.savefig(os.path.join(save_folder, f'task{task}_pic{pic}_frame.png'))
        # plt.close()
        plot_feature_difference_paper_selection(data_all, subject_HC, subject_MCI, subject_Severe, task, pic, pic_frame, save_path)

if __name__ == '__main__':
    select_pic_list = [
        ['023_patient', '067_patient', '043_patient', 0, 3],
        ['023_patient', '067_patient', '043_patient', 1, 0],
        ['023_patient', '067_patient', '043_patient', 2, 3],
        ['023_patient', '048_patient', '024_patient', 3, 0],
        ['023_patient', '048_patient', '024_patient', 4, 0],
        ['023_patient', '048_patient', '024_patient', 5, 3],
        ['023_patient', '048_patient', '024_patient', 6, 4],
        ['023_patient', '048_patient', '024_patient', 7, 3],
        ['023_patient', '048_patient', '032_patient', 8, 4],
    ]
    main_select_pic(select_pic_list, sliced_path, video_path, save_path)

