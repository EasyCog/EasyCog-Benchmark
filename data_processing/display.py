import numpy as np
import matplotlib as plt

from data_processing.analysis_utils import *


def display_exg(data, sps=125.0, mode='parallel', plot='plot', spectrum=False, savepath=None):
    """
    data: [n_ch, n_samples],
    sps: int, sample rates,
    mode: 'overlap', 'parallel',
    plot: 'plot', 'save',
    """
    if mode == 'parallel':
        if spectrum:
            fft_data = np.fft.fft(data, axis=1)
            fft_data = np.abs(fft_data)
            fft_data = fft_data / data.shape[1]
            # fft_data = fft_data[0:data.shape[1]//2+1]   
            fft_freq = np.fft.fftfreq(data.shape[1], d=1/sps)
            plt.figure()
            plt.xlabel('Frequency(Hz)')
            plt.ylabel('Amplitude')
            plt.title('Spectrum')
            for i in range(data.shape[0]):
                plt.subplot(data.shape[0], 1, i+1)
                plt.plot(fft_freq, fft_data[i])
                plt.ylabel(STD_CH[i])
                plt.xlim([0, sps/2])
            plt.xlabel('Frequency(Hz)')
            if plot == 'plot':
                plt.show()
            else:
                plt.savefig(savepath)
        else:
            t = np.arange(data.shape[1])/sps
            plt.figure(figsize=(8,15))
            # X_factor = np.mean(np.std(data, axis=1))*10
            # plt.plot(t, data.T + np.arange(-8, 8) * X_factor)
            # # plt.xlim([TASK_7_START,TASK_7_END])
            # plt.xlabel('time(sec)')
            # plt.yticks(np.arange(-8, 8)*X_factor,STD_CH)
            # plt.ylim([-9*X_factor, 8*X_factor])
            for i in range(data.shape[0]):
                plt.subplot(data.shape[0], 1, i+1)
                plt.plot(t, data[i])
                plt.ylabel(STD_CH[i])

            plt.xlabel('time(sec)')
            if plot == 'plot':
                plt.show()
            else:
                plt.savefig(savepath)
    elif mode == 'overlap':
        if spectrum:
            fft_data = np.fft.fft(data, axis=1)
            fft_data = np.abs(fft_data)
            fft_data = fft_data / data.shape[1]
            fft_data = fft_data[0:data.shape[1]//2+1]
            fft_freq = np.linspace(0, sps/2, fft_data.shape[1]) 
            plt.figure()
            plt.xlabel('Frequency(Hz)')
            plt.ylabel('Amplitude')
            plt.title('Spectrum')
            plt.plot(fft_freq, fft_data)
            if plot == 'plot':
                plt.show()
            else:
                plt.savefig(savepath)
        else:
            t = np.arange(data.shape[1])/sps
            plt.figure()
            for i in range(data.shape[0]):
                plt.plot(t, data[i])
            # plt.xlim([TASK_7_START,TASK_7_END])
            plt.xlabel('time(sec)')
            if plot == 'plot':
                plt.show()
            else:
                plt.savefig(savepath)
    else:
        raise NotImplementedError
    # for i in range(len(eye_mov_start_point)):
    #     plt.axvline(x=eye_mov_start_point[i], color='r', linestyle='--')