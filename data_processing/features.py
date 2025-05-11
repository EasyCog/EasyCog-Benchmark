"""
Feature extraction module for EEG signal processing.

This module provides a comprehensive set of features for EEG signal analysis,
including frequency domain, time domain, and spatial features.
"""

import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew, iqr
from sklearn.cluster import KMeans
from statsmodels.tsa.vector_ar.var_model import VAR
from time import time
from typing import List, Tuple, Dict, Union, Optional
import os
from sklearn.decomposition import PCA

FEATURE_NAMES = [
    'delta_ratio', 'theta_ratio', 'alpha_ratio', 'beta_ratio', 'gamma_ratio',
    'beta_to_alpha_ratio', 'theta_to_alpha_ratio', 'theta_alpha_to_beta_ratio',
    'theta_to_beta_ratio', 'theta_alpha_to_alpha_beta_ratio', 'gamma_to_delta_ratio',
    'gamma_beta_to_delta_alpha_ratio', 'frequency_spectrum', 'stft_spectrum',
    'mean', 'std', 'median', 'micro_states', 'signal_complexity', 'entropy',
    'channel_correlation', 'mutual_information', 'peak_value', 'rectification_avg',
    'effecteive_value', 'kurtosis', 'skewness', 'margin', 'form_factor', 'impulse_factor','crest_factor',
    'micro_states_occurrences', 'micro_states_transition', 'micro_states_dist', 
      'micro_states_center',
]

class FeatureGenerator:
    """
    A class for generating various features from EEG signals.
    
    This class provides methods to extract different types of features from EEG data,
    including frequency domain features, time domain features, and spatial features.
    
    Attributes:
        data (np.ndarray): Input EEG data with shape (n_channels, n_samples)
        sps (int): Samples per second (sampling rate)
    """
    
    def __init__(self, data: np.ndarray, sps: int = 125) -> None:
        """
        Initialize the FeatureGenerator.
        
        Args:
            data: EEG data array with shape (n_channels, n_samples)
            sps: Sampling rate in Hz (default: 125)
        """
        self.data = data
        self.sps = sps
    
    ### Frequency band power ratios
    def delta_ratio(self):
        delta_band = self._bandpower(self.data, 0.5, 4)
        total_power = self._bandpower(self.data, 0, 64)
        return delta_band / total_power
    
    def theta_ratio(self):
        theta_band = self._bandpower(self.data, 4, 8)
        total_power = self._bandpower(self.data, 0, 64)
        return theta_band / total_power
    
    def alpha_ratio(self):
        alpha_band = self._bandpower(self.data, 8, 12)
        total_power = self._bandpower(self.data, 0, 64)
        return alpha_band / total_power
    
    def beta_ratio(self):
        beta_band = self._bandpower(self.data, 12, 30)
        total_power = self._bandpower(self.data, 0, 64)
        return beta_band / total_power
    
    def gamma_ratio(self):
        gamma_band = self._bandpower(self.data, 30, 64)
        total_power = self._bandpower(self.data, 0, 64)
        return gamma_band / total_power
    
    def beta_to_alpha_ratio(self):
        beta_band = self._bandpower(self.data, 12, 30)
        alpha_band = self._bandpower(self.data, 8, 12)
        return beta_band / alpha_band
    
    def theta_to_alpha_ratio(self):
        theta_band = self._bandpower(self.data, 4, 8)
        alpha_band = self._bandpower(self.data, 8, 12)
        return theta_band / alpha_band
    
    def theta_alpha_to_beta_ratio(self):
        theta_alpha_band = self._bandpower(self.data, 4, 12)
        beta_band = self._bandpower(self.data, 12, 30)
        return theta_alpha_band / beta_band

    def theta_to_beta_ratio(self):
        theta_band = self._bandpower(self.data, 4, 8)
        beta_band = self._bandpower(self.data, 12, 30)
        return theta_band / beta_band
    
    def theta_alpha_to_alpha_beta_ratio(self):
        theta_alpha_band = self._bandpower(self.data, 4, 12)
        alpha_beta_band = self._bandpower(self.data, 8, 30)
        return theta_alpha_band / alpha_beta_band
    
    def gamma_to_delta_ratio(self):
        gamma_band = self._bandpower(self.data, 30, 64)
        delta_band = self._bandpower(self.data, 0.5, 4)
        return gamma_band / delta_band
    
    def gamma_beta_to_delta_alpha_ratio(self):
        gamma_beta_band = self._bandpower(self.data, 12, 64)
        delta_band = self._bandpower(self.data, 0.5, 4)
        alpha_band = self._bandpower(self.data, 8, 12)
        return (gamma_beta_band) / (delta_band + alpha_band)

    def frequency_spectrum(self):
        # Compute the frequency spectrum
        freqs, psd = self._compute_psd(self.data)
        return psd
    
    def stft_spectrum(self, window=1, step=0.5):
        # Compute the Short-Time Fourier Transform (STFT) spectrum
        window, step = int(window * self.sps), int(step * self.sps)
        f, t, Zxx = signal.stft(self.data, fs=self.sps, window='hann', nperseg=window, noverlap=step)
        return Zxx
    
    def _bandpower(self, data, low, high):
        # Calculate the power of the frequency band
        freqs, psd = self._compute_psd(data)
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        bandpower = np.trapz(psd[:, idx_band], freqs[idx_band], axis=1)
        return bandpower
    
    def _compute_psd(self, data):
        # Compute the Power Spectral Density (PSD)
        freqs = np.fft.rfftfreq(data.shape[1], d=1./self.sps)
        psd = np.abs(np.fft.rfft(data, axis=1))**2
        return freqs, psd

    ## Time-domain
    def mean(self):
        return np.mean(self.data, axis=1)
    
    def std(self):
        return np.std(self.data, axis=1)
    
    def median(self):
        return np.median(self.data, axis=1)
    
    def micro_states(self):
        return self._micro_states(self.data)
    
    def _micro_states(self, data, l = 0.3):
        # data: [n_ch, n_sample]
        # Divide the data into 100ms segments and use
        # STD for each segment as the micro state for
        # segments. Then re-map the data into the
        # sequence of micro states.
        
        segment_length = int(l * self.sps)  # 100ms segments
        n_channels, n_samples = data.shape
        n_segments = n_samples // segment_length
        
        micro_states = np.zeros((n_channels, n_segments))
        
        for ch in range(n_channels):
            for seg in range(n_segments):
                segment_data = data[ch, seg * segment_length:(seg + 1) * segment_length]
                micro_states[ch, seg] = np.std(segment_data)
        
        return micro_states

    def micro_states_occurrences(self):
        return self._micro_states_occurrences(self.data)

    def signal_complexity(self):
        return self._signal_complexity(self.data)
    
    def _signal_complexity(self, data):
        # Compute the signal Lempel-Ziv complexity
        n_channels = data.shape[0]
        complexity = np.zeros(n_channels)
        for i in range(n_channels):
            complexity[i] = self._lempel_ziv_complexity(data[i])
        return complexity

    def _lempel_ziv_complexity(self, data, l=10):
        # Compute the Lempel-Ziv complexity of the signal.
        # Specifically, define l different thresholds of amplitude
        # (given by max and min of amplitude), the first range of amplitude
        # is mapped to 0, the second range is mapped to 1, and so on.
        # Then the data is mapped into the new sequence of integers.
        # After that, the Lempel-Ziv complexity is calculated.
        n_sample = data.shape

        # Normalize the data
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Quantize the data into l levels
        quantized_data = np.floor(data * l).astype(int)
        
        # Convert the quantized data to a binary sequence
        binary_sequence = ''.join(format(x, 'b').zfill(int(np.log2(l))) for x in quantized_data)
        
        # Calculate the Lempel-Ziv complexity
        i, k, lzc = 0, 1, 1
        while k < len(binary_sequence):
            if binary_sequence[i:k] not in binary_sequence[:i]:
                lzc += 1
                i = k
            k += 1
        
        # lzc is the log 

        return lzc*(np.log(n_sample)/np.log(l)) / n_sample

    # fractal dimension
    def peak_value(self):
        return np.max(self.data, axis=1)-np.min(self.data, axis=1)
    
    def rectification_avg(self):
        return np.mean(np.abs(self.data), axis=1)
    
    def effecteive_value(self):
        return np.sqrt(np.mean(self.data**2, axis=1))
    
    def kurtosis(self):
        return kurtosis(self.data, axis=1)
    
    def skewness(self):
        return skew(self.data, axis=1)
    
    def margin(self):
        return self.peak_value()/np.mean(np.sqrt(np.abs(self.data)), axis=1)**2

    def form_factor(self):
        return self.effecteive_value()/self.rectification_avg()

    def impulse_factor(self):
        return self.peak_value()/self.rectification_avg()

    def crest_factor(self):
        return self.peak_value()/self.effecteive_value()    

    ## entropy
    def entropy(self):
        return self._entropy(self.data)
    
    def _entropy(self, data):
        # Compute the entropy of the data for each channel
        n_channels = data.shape[0]
        entropy = np.zeros(n_channels)
        for i in range(n_channels):
            p = np.histogram(data[i], bins=10, density=True)[0]
            entropy[i] = -np.sum(p * np.log2(p + 1e-6))

        return entropy

    ## spatial
    def directed_transfer_function(self):
        return self._directed_transfer_function(self.data)
    
    def _directed_transfer_function(self, data, freq_range=(0, 62), n_freqs=100, order=5, sps=125):
        # Compute the directed transfer function between channels
        n_channels = data.shape[0]
        
        # fit VAR model
        model = VAR(data.T)
        try:
            results = model.fit(order)
        except Exception as e:
            print(f"Error fitting VAR model: {e}")
            return np.zeros((n_channels, n_channels, n_freqs))

        # get MVAR coefficients
        A = results.coefs.transpose((1, 2, 0))

        # define frequency_range
        freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)

        # compute DTF
        dtf = np.zeros((n_channels, n_channels, n_freqs))
        for i, f in enumerate(freqs):
            z = np.exp(-2j * np.pi * f / sps)
            H = np.eye(n_channels, dtype=complex)
            for k in range(order):
                H -= A[:, :, k] * z** (k + 1)
            H_inv = np.linalg.inv(H)
            for j in range(n_channels):
                dtf[:, j, i] = np.abs(H_inv[:, j])**2 / np.sum(np.abs(H_inv)**2, axis=1)

        return dtf

    def phase_locking_value(self):
        return self._phase_locking_value(self.data)

    def _phase_locking_value(self, data):
        # Compute the phase locking value between channels
        n_channels = data.shape[0]
        plv = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    plv[i, j] = self._compute_plv(data[i], data[j])
        return plv
    
    def _compute_plv(self, x, y):
        # Compute the phase locking value between signals x and y
        phase_x = np.angle(signal.hilbert(x))
        phase_y = np.angle(signal.hilbert(y))
        plv = np.abs(np.mean(np.exp(1j * (phase_x - phase_y))))
        return plv

    def phase_lag_index(self):
        return self._phase_lag_index(self.data)
    
    def _phase_lag_index(self, data):
        # Compute the phase lag index between channels
        n_channels = data.shape[0]
        pli = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    pli[i, j] = self._compute_pli(data[i], data[j])
        return pli
    
    def _compute_pli(self, x, y):
        # Compute the phase lag index between signals x and y
        phase_x = np.angle(signal.hilbert(x))
        phase_y = np.angle(signal.hilbert(y))
        pli = np.abs(np.mean(np.sign(phase_x - phase_y)))
        return pli


    def transfer_entropy(self):
        return self._transfer_entropy(self.data)
    
    def _transfer_entropy(self, data):
        # Compute the transfer entropy between channels
        n_channels = data.shape[0]
        te = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    te[i, j] = self._compute_te(data[i], data[j])
                    # te[i, j] = te_compute.te_compute(data[i], data[j], 1, 1, safetyCheck=False, GPU=False)
        return te
    
    def _compute_te(self, X, Y, delay=1):
        # Compute the transfer entropy between signals x and y in fast way
        if len(X)!=len(Y):
            raise ValueError('time series entries need to have same length')

        n = float(len(X[delay:]))

        # number of bins for X and Y using Freeman-Diaconis rule
        # histograms built with numpy.histogramdd
        binX = int( (max(X)-min(X))
                    / (2* iqr(X) / (len(X)**(1.0/3))) )
        binY = int( (max(Y)-min(Y))
                    / (2* iqr(Y) / (len(Y)**(1.0/3))) )

        # Definition of arrays of shape (D,N) to be transposed in histogramdd()
        x3 = np.array([X[delay:],Y[:-delay],X[:-delay]])
        x2 = np.array([X[delay:],Y[:-delay]])
        x2_delay = np.array([X[delay:],X[:-delay]])

        p3,bin_p3 = np.histogramdd(
            sample = x3.T,
            bins = [binX,binY,binX])

        p2,bin_p2 = np.histogramdd(
            sample = x2.T,
            bins=[binX,binY])

        p2delay,bin_p2delay = np.histogramdd(
            sample = x2_delay.T,
            bins=[binX,binX])

        p1,bin_p1 = np.histogramdd(
            sample = np.array(X[delay:]),
            bins=binX)

        # Hists normalized to obtain densities
        p1 = p1/n
        p2 = p2/n
        p2delay = p2delay/n
        p3 = p3/n

        # Ranges of values in time series
        Xrange = bin_p3[0][:-1]
        Yrange = bin_p3[1][:-1]
        X2range = bin_p3[2][:-1]

        # Calculating elements in TE summation
        elements = []
        for i in range(len(Xrange)):
            px = p1[i]
            for j in range(len(Yrange)):
                pxy = p2[i][j]

                for k in range(len(X2range)):
                    pxx2 = p2delay[i][k]
                    pxyx2 = p3[i][j][k]

                    arg1 = float(pxy*pxx2)
                    arg2 = float(pxyx2*px)

                    # Corrections avoding log(0)
                    if arg1 == 0.0: arg1 = float(1e-8)
                    if arg2 == 0.0: arg2 = float(1e-8)

                    term = pxyx2*np.log2(arg2) - pxyx2*np.log2(arg1) 
                    elements.append(term)

        # Transfer Entropy
        TE = np.sum(elements)
        return TE

    def coherence(self):
        return self._coherence(self.data)

    def _coherence(self, data):
        # Compute the coherence between channels
        n_channels = data.shape[0]
        coherence_matrix = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    f, Cxy = signal.coherence(data[i], data[j], fs=self.sps)
                    coherence_matrix[i, j] = np.mean(Cxy)
        return coherence_matrix

    def channel_correlation(self):
        return np.corrcoef(self.data)
    
    def mutual_information(self):
        return self._mutual_information(self.data)
    
    def _mutual_information(self, data):
        # Compute the mutual information between channels
        n_channels = data.shape[0]
        mi = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                mi[i, j] = self._information(np.vstack([data[i], data[j]]))
        return mi

    def _information(self, data):
        hist, _ = np.histogram(data, bins=100)
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-12))
        return entropy

    def calculate_feature(self, feature_name):
        if hasattr(self, feature_name):
            return getattr(self, feature_name)()
        else:
            raise ValueError(f"Feature '{feature_name}' is not defined.")

def cluster_microstate(data, l=0.3, n_clusters=20):
    # Cluster the microstates
    feature_gen = FeatureGenerator(data)
    micro_states = feature_gen._micro_states(data, l)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(micro_states.T)
    return kmeans.labels_, kmeans.cluster_centers_

def micro_state_mapping(microstate, microstate_centers):
    # Map the data into the sequence of microstates
    n_channels, n_microstate = microstate.shape
    n_microstates = microstate_centers.shape[0]
    microstate_sequence = np.zeros(n_microstate, dtype=int)

    for i in range(n_microstate):
        segment_data = microstate[:, i]
        distances = np.linalg.norm(microstate_centers - segment_data[np.newaxis, :], axis=0)
        microstate_sequence[i] = np.argmin(distances)

    return microstate_sequence


# Example usage:
# data = np.random.randn(1000)  # Example data
# feature_gen = FeatureGenerator(data)
# feature_value = feature_gen.calculate_feature('delta_to_beta_ratio')
# print(feature_value)

def calculate_pca_feat(sliced_feat_path, processed_sliced_feat_train_path):
    cnt = 0
    all_features = []
    files_name = []
    sliced_files = os.listdir(sliced_feat_path)
    
    # Load and process features
    for file in sliced_files:
        cnt += 1
        # print the progress
        if cnt % 500 == 0:
            print(f"Loaded {cnt}/{len(sliced_files)} files")

        filepath = os.path.join(sliced_feat_path, file)
        processed_feat_file_path = os.path.join(processed_sliced_feat_train_path, 
                                              f"{file.split('.')[0]}-feat.npz")
        
        if os.path.isfile(filepath):
            unsliced_data = np.load(processed_feat_file_path, allow_pickle=True)
            files_name.append(file)
            combined_features = []

            for f in FEATURE_NAMES:
                if f in ['frequency_spectrum', 'stft_spectrum', 'micro_states']:
                    continue
                value = unsliced_data[f]
                if np.isnan(value).any():
                    print(f"File {file} has nan value in feature {f}")
                combined_features.append(value.flatten())
            
            combined_features = np.concatenate(combined_features)        
            all_features.append(combined_features)

    # Perform PCA
    all_features = np.array(all_features)
    pca = PCA(n_components=8)
    reduced_features = pca.fit_transform(all_features)

    # Save reduced features
    cnt = 0  
    for file in files_name:
        cnt += 1
        if cnt % 500 == 0:
            print(f"Saved PCA features to {cnt}/{len(files_name)} files")
            
        processed_feat_file_path = os.path.join(processed_sliced_feat_train_path, 
                                              f"{file.split('.')[0]}-feat.npz")
        unsliced_data = dict(np.load(processed_feat_file_path, allow_pickle=True))
        unsliced_data['pca_reduced_features'] = reduced_features[files_name.index(file)]
        np.savez(processed_feat_file_path, **unsliced_data)

def read_sliced_feat(path):
    data = np.load(path, allow_pickle=True)
    eeg = data["eeg_seg"]
    eog = data["eog_seg"]
    et = data["et_seg"]
    moca = data["moca"]
    mmse = data["mmse"]
    task_score = data["moca_task_score"]
    mmse_task_score = data["mmse_task_score"]
    eeg_std_all = data["eeg_std_all"]
    eog_mean_all = data["eog_mean_all"]
    eeg_mean_all = data["eeg_mean_all"]
    eog_std_all = data["eog_std_all"]
    return eeg, eog, et, moca, mmse, task_score, mmse_task_score, eeg_std_all, eog_mean_all, eeg_mean_all, eog_std_all


def test_feature_generator():
    # Generate example data with 8 channels and 1000 samples per channel
    # create signal with different frequency bands energy
    delta_amp, theta_amp, alpha_amp, beta_amp, gamma_amp = 1, 2, 3, 4, 5
    delta_freq, theta_freq, alpha_freq, beta_freq, gamma_freq = 1, 5, 10, 20, 40

    t = np.linspace(0, 10, 1250)
    delta = delta_amp * np.sin(2 * np.pi * delta_freq * t)
    theta = theta_amp * np.sin(2 * np.pi * theta_freq * t)
    alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t)
    beta = beta_amp * np.sin(2 * np.pi * beta_freq * t)
    gamma = gamma_amp * np.sin(2 * np.pi * gamma_freq * t)

    data = delta+theta+alpha+beta+gamma
    eight_channel_noise = np.random.randn(8, 1250)
    data = np.vstack([data]*8) + eight_channel_noise
    feature_gen = FeatureGenerator(data)

    # Test frequency band power ratios
    print("Delta Ratio:", feature_gen.calculate_feature('delta_ratio'))
    print("Theta Ratio:", feature_gen.calculate_feature('theta_ratio'))
    print("Alpha Ratio:", feature_gen.calculate_feature('alpha_ratio'))
    print("Beta Ratio:", feature_gen.calculate_feature('beta_ratio'))
    print("Gamma Ratio:", feature_gen.calculate_feature('gamma_ratio'))
    print("Beta to Alpha Ratio:", feature_gen.calculate_feature('beta_to_alpha_ratio'))
    print("Theta to Alpha Ratio:", feature_gen.calculate_feature('theta_to_alpha_ratio'))
    print("Theta Alpha to Beta Ratio:", feature_gen.calculate_feature('theta_alpha_to_beta_ratio'))
    print("Theta to Beta Ratio:", feature_gen.calculate_feature('theta_to_beta_ratio'))
    print("Theta Alpha to Alpha Beta Ratio:", feature_gen.calculate_feature('theta_alpha_to_alpha_beta_ratio'))
    print("Gamma to Delta Ratio:", feature_gen.calculate_feature('gamma_to_delta_ratio'))
    print("Gamma Beta to Delta Alpha Ratio:", feature_gen.calculate_feature('gamma_beta_to_delta_alpha_ratio'))

    # Test frequency spectrum
    print("Frequency Spectrum:", feature_gen.calculate_feature('frequency_spectrum'))

    # Test STFT spectrum
    print("STFT Spectrum:", feature_gen.calculate_feature('stft_spectrum'))

    # Test time-domain features
    print("Mean:", feature_gen.calculate_feature('mean'))
    print("STD:", feature_gen.calculate_feature('std'))
    print("Median:", feature_gen.calculate_feature('median'))
    print("Micro States:", feature_gen.calculate_feature('micro_states'))

    # Test signal complexity
    print("Signal Complexity:", feature_gen.calculate_feature('signal_complexity'))

    # Test entropy
    print("Entropy:", feature_gen.calculate_feature('entropy'))

    # Test spatial features
    print("Channel Correlation:", feature_gen.calculate_feature('channel_correlation'))
    print("Mutual Information:", feature_gen.calculate_feature('mutual_information'))
    print("Directed Information Flow:", feature_gen.calculate_feature('directed_information_flow'))
    print("Phase Locking Value:", feature_gen.calculate_feature('phase_locking_value'))
    print("Coherence:", feature_gen.calculate_feature('coherence'))

    # Test fractal features
    print("Peak Value:", feature_gen.calculate_feature('peak_value'))
    print("Rectification Average:", feature_gen.calculate_feature('rectification_avg'))
    print("Effective Value:", feature_gen.calculate_feature('effecteive_value'))
    print("Kurtosis:", feature_gen.calculate_feature('kurtosis'))
    print("Skewness:", feature_gen.calculate_feature('skewness'))
    print("Margin:", feature_gen.calculate_feature('margin'))
    print("Form Factor:", feature_gen.calculate_feature('form_factor'))
    print("Impulse Factor:", feature_gen.calculate_feature('impulse_factor'))
    print("Crest Factor:", feature_gen.calculate_feature('crest_factor'))

if __name__ == "__main__":
    ## test feature generator
    # test_feature_generator()


    # test data loading
    # data = np.load('features/processed_sliced_feat/ASR_ASR_EOG_resampleET/030_patient-2024_12_11_17_31_07-resting-task0-pic0-7375-7750-feat.npz', allow_pickle=True)
    delta_amp, theta_amp, alpha_amp, beta_amp, gamma_amp = 1, 2, 3, 4, 5
    delta_freq, theta_freq, alpha_freq, beta_freq, gamma_freq = 1, 5, 10, 20, 40

    t = np.linspace(0, 10, 125*3)
    delta = delta_amp * np.sin(2 * np.pi * delta_freq * t)
    theta = theta_amp * np.sin(2 * np.pi * theta_freq * t)
    alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t)
    beta = beta_amp * np.sin(2 * np.pi * beta_freq * t)
    gamma = gamma_amp * np.sin(2 * np.pi * gamma_freq * t)

    data = delta+theta+alpha+beta+gamma
    eight_channel_noise = np.random.randn(16, 125*3)
    data = np.vstack([data]*16) + eight_channel_noise
    feature_gen = FeatureGenerator(data)
    total_start = time()
    for i in FEATURE_NAMES:
        if 'micro' in i:
            continue
        # time each feature
        start = time()
        feat = feature_gen.calculate_feature(i)
        print(f'{i}: {feat} - {time()-start}')

    print("Total time:", time()-total_start)


