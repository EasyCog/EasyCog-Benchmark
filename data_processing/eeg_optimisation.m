function eeg_cleaned = eeg_optimisation(eeg,fs)
% SJN 2025/02/25
[eeg_channels, eeg_length] = size(eeg); %高通
[b_hp, a_hp] = butter(8, 1 / (fs / 2), 'high'); % 8阶 IIR
eeg_highpass = filtfilt(b_hp, a_hp, double(eeg'))';
filter_order_bp = round(3 * fs); % 带通 FIR 滤波 (1-45 Hz) 去除噪声
b_bp = fir1(filter_order_bp, [1, 45] / (fs / 2), 'bandpass');
eeg_referenced = filtfilt(b_bp, 1, eeg_highpass')';
%%%%%ASR 去伪影%%%
EEG = pop_importdata('dataformat', 'array', 'nbchan', eeg_channels, ...
                     'data', eeg_referenced, 'srate', fs);
EEG_clean = clean_rawdata(EEG, 'off', 'off','off','off', 100, -1);
eeg_cleaned_raw = EEG_clean.data;
eeg_length_cleaned = size(eeg_cleaned_raw, 2);
%%%%%插值修复缺失点%%%%%%%
if eeg_length_cleaned < eeg_length
    eeg_cleaned = interp1(linspace(1, eeg_length_cleaned, eeg_length_cleaned), ...
                          eeg_cleaned_raw', linspace(1, eeg_length_cleaned, eeg_length))';
else
    eeg_cleaned = eeg_cleaned_raw;
end
end