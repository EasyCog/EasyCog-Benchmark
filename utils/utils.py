import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random
import yaml
from matplotlib.patches import Circle, Polygon
from scipy.interpolate import griddata
from scipy.signal import butter, sosfiltfilt
import json
from utils import logger
import utils.balanced_subject_split as balanced_subject_split
import argparse
import torch
import pynvml
from data_processing.excel_operation import read_xlsx_to_dict

def get_frequency_component(data, frequency='alpha', fs=500, axis=0):
	# data = dataset[subject][electrode]
	if frequency == 'alpha':
		# [8, 12]
		ft = butter(10, [8, 12], 'bp', fs=fs, output='sos')
	elif frequency == 'sigma':
		# [12, 16]
		ft = butter(10, [12, 16], 'bp', fs=fs, output='sos')
	elif frequency == 'beta':
		# [13, 30]
		ft = butter(10, [13, 30], 'bp', fs=fs, output='sos')
	elif frequency == 'delta':
		# [0.5, 4]
		ft = butter(10, [0.3, 4], 'bp', fs=fs, output='sos')        
	elif frequency == 'gamma':
		# [30, 100]
		ft = butter(10, [30, 50], 'bp', fs=fs, output='sos')
	elif frequency == 'theta':
		# [4, 7]
		ft = butter(10, [4, 7], 'bp', fs=fs, output='sos')
	filtered_data = sosfiltfilt(ft, data, axis=axis)
	return filtered_data

def compute_energy(data):
	return np.sum(np.abs(data)**2)

def generate_save_file_folder(save_path):
	return os.path.join(save_path, get_now_time())

def get_now_time():
	return time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))

def make_data_folder(root_dir, subject_name):
	subject_dir = os.path.join(root_dir, subject_name)
	subject_dir_time = generate_save_file_folder(subject_dir)
	os.makedirs(subject_dir_time, exist_ok=True)
	return subject_dir_time



# FR1~FR4 is the frontal channels (from right to left 1-4), R1~R10 is the right channels, L1~L10 is the left channels
CH_NAMES = ['R1', 'R2', 'R3', 'FR2', 'FR1', 'R8', 'R9', 'R10', 'L1', 'L2', 'L3', 'FR4', 'FR3', 'L8', 'L9', 'L10']   # 头带出线口在下侧/脸右侧
CH_NAMES2 = ['R1', 'R2', 'R3', 'FR3', 'FR4', 'R8', 'R9', 'R10', 'L1', 'L2', 'L3', 'FR1', 'FR2', 'L8', 'L9', 'L10']   # 头带出线口在上侧/脸左侧

STD_CH = ['R1', 'R2', 'R3', 'FR1', 'FR2', 'R8', 'R9', 'R10', 'L1', 'L2', 'L3', 'FR4', 'FR3', 'L8', 'L9', 'L10']     # 标准的通道顺序

FPS = 30
TASK_DURATION = (430*FPS+2)/FPS
TASK_1_START = (4*FPS+2)/FPS
TASK_1_PIC_START = [4*FPS+2, 9*FPS, 14*FPS, 19*FPS, 24*FPS, 29*FPS, 34*FPS, 39*FPS, 44*FPS, 49*FPS]
TASK_1_PIC_START = [int(i/FPS) for i in TASK_1_PIC_START]
TASK_1_END = (54*FPS)/FPS
TASK_2_START = (56*FPS)/FPS
TASK_2_PIC_START = [56*FPS, 61*FPS, 66*FPS, 71*FPS, 76*FPS, 81*FPS, 86*FPS, 91*FPS, 96*FPS, 101*FPS]
TASK_2_PIC_START = [int(i/FPS) for i in TASK_2_PIC_START]
TASK_2_END = (106*FPS)/FPS
TASK_3_START = (107*FPS+29)/FPS
TASK_3_PIC_START = [107*FPS+29, 112*FPS+29, 117*FPS+29, 122*FPS+29, 127*FPS+29, 132*FPS+29, 137*FPS+29, 142*FPS+29, 147*FPS+29, 152*FPS+29]
TASK_3_PIC_START = [int(i/FPS) for i in TASK_3_PIC_START]
TASK_3_END = (157*FPS+29)/FPS
TASK_4_START = (159*FPS+29)/FPS
TASK_4_PIC_START = [159*FPS+29, 164*FPS+29, 169*FPS+29, 174*FPS+29, 179*FPS+29, 184*FPS+29, 189*FPS+29, 194*FPS+29, 199*FPS+29, 204*FPS+29]
TASK_4_PIC_START = [int(i/FPS) for i in TASK_4_PIC_START]
TASK_4_END = (209*FPS+29)/FPS
TASK_5_START = (211*FPS+29)/FPS
TASK_5_PIC_START = [211*FPS+29, 216*FPS+29, 221*FPS+29, 226*FPS+29, 231*FPS+29, 236*FPS+29, 241*FPS+29, 246*FPS+29, 251*FPS+29, 256*FPS+29]
TASK_5_PIC_START = [int(i/FPS) for i in TASK_5_PIC_START]
TASK_5_END = (261*FPS+29)/FPS
TASK_6_START = (263*FPS+29)/FPS
TASK_6_PIC_START = [263*FPS+29, 268*FPS+29, 273*FPS+29, 279*FPS+2, 284*FPS+2, 289*FPS+2]
TASK_6_PIC_START = [int(i/FPS) for i in TASK_6_PIC_START]
TASK_6_END = (294*FPS+2)/FPS
TASK_7_START = (296*FPS+2)/FPS
TASK_7_PIC_START = [296*FPS+2, 301*FPS+2, 306*FPS+2, 311*FPS+2, 316*FPS+2, 321*FPS+2, 326*FPS+2, 331*FPS+2, 336*FPS+2, 341*FPS+2]
TASK_7_PIC_START = [int(i/FPS) for i in TASK_7_PIC_START]
TASK_7_END = (346*FPS+2)/FPS
TASK_8_START = (348*FPS+2)/FPS
TASK_8_PIC_START = [348*FPS+2, 353*FPS+2, 358*FPS+2, 363*FPS+2, 368*FPS+2, 373*FPS+2, 378*FPS+2, 383*FPS+2, 388*FPS+2, 393*FPS+2]
TASK_8_PIC_START = [int(i/FPS) for i in TASK_8_PIC_START]
TASK_8_END = (398*FPS+2)/FPS
TASK_9_START = (400*FPS+2)/FPS
TASK_9_PIC_START = [400*FPS+2, 403*FPS+2, 406*FPS+2, 409*FPS+2, 412*FPS+2, 415*FPS+2, 418*FPS+2, 421*FPS+2, 424*FPS+2, 427*FPS+2]
TASK_9_PIC_START = [int(i/FPS) for i in TASK_9_PIC_START]
TASK_9_END = (430*FPS+2)/FPS

def read_bci_txt(file_path):
	"""
	Read a BCI Competition IV dataset from a .txt file.
	
	Parameters
	----------
	file_path : str
		The path to the .txt file.
	
	Returns
	-------
	data : numpy.ndarray
		The data contained in the .txt file.
	"""
	# Read the data from the .txt file
	with open(file_path, 'r') as file:
		data = file.readlines()
	
	for i in range(len(data)-1, -1, -1):
		if data[i].startswith('%') or data[i].startswith('S'):
			data.pop(i)

	# Split the data into a list of strings
	data = [line.split(', ') for line in data]
	data = [line[:-1] for line in data]

	# Convert the strings to floats
	data = np.array(data, dtype=float)

	return data

def read_config(cfg_file):
	cfg = yaml.load(open(cfg_file), Loader=yaml.FullLoader)
	return cfg


def read_data(path):
	data = np.load(os.path.join(path, 'exg_data.npz'))
	exg_data = data['data'][data['exg_channels'],:]
	exg_timestamp = data['data'][data['timestamp_channel'],:]
	sampling_rate = float(data['sampling_rate'])
	camera_timestamps_webcam = np.load(os.path.join(path, 'recorded_video_frame_timestamps_in_record_webcam_function.npy'))
	camera_timestamps = np.loadtxt(os.path.join(path, 'recorded_video_frame_timestamps.txt'))
	start_points = np.load(os.path.join(path, 'timestamps.npz'))

	return exg_data, exg_timestamp, sampling_rate, start_points, camera_timestamps_webcam, camera_timestamps

def read_data_debug(path):
	data = np.load(os.path.join(path, 'exg_data.npz'))
	exg_data = data['data'][data['exg_channels'],:]
	exg_timestamp = data['data'][data['timestamp_channel'],:]
	sampling_rate = float(data['sampling_rate'])
	camera_timestamps_webcam = np.load(os.path.join(path, 'recorded_video_frame_timestamps_in_record_webcam_function.npy'))
	camera_timestamps = np.loadtxt(os.path.join(path, 'recorded_video_frame_timestamps.txt'))

	return exg_data, exg_timestamp, sampling_rate, camera_timestamps_webcam, camera_timestamps

def process_packet_loss(exg_data, exg_timestamps, sps):
	return None

def split_tasks_exg(exg_data, sps, video_duration, timestamp=None):
	real_fps_scale = TASK_DURATION/video_duration
	sps = int(sps / real_fps_scale)  # if video play faster, then the relative sps will be smaller to the video
	"""
	E.g., normally when video play 30 frames, 1s pass by and sps samples are collected, however
	if video play 60 frames per second (faster than normally), than only sps/2 samples are collected
	"""

	# if exg_data.shape[1] < TASK_DURATION*sps:
	#     return None
	
	task_data = {}
	task_data['task1'] = exg_data[:, int(TASK_1_START*sps):int(TASK_1_END*sps)]
	task_data['task2'] = exg_data[:, int(TASK_2_START*sps):int(TASK_2_END*sps)]
	task_data['task3'] = exg_data[:, int(TASK_3_START*sps):int(TASK_3_END*sps)]
	task_data['task4'] = exg_data[:, int(TASK_4_START*sps):int(TASK_4_END*sps)]
	task_data['task5'] = exg_data[:, int(TASK_5_START*sps):int(TASK_5_END*sps)]
	task_data['task6'] = exg_data[:, int(TASK_6_START*sps):int(TASK_6_END*sps)]
	task_data['task7'] = exg_data[:, int(TASK_7_START*sps):int(TASK_7_END*sps)]
	task_data['task8'] = exg_data[:, int(TASK_8_START*sps):int(TASK_8_END*sps)]
	task_data['task9'] = exg_data[:, int(TASK_9_START*sps):int(TASK_9_END*sps)]

	task_data['task1_pic'] = []
	for i in range(len(TASK_1_PIC_START)-1):
		task_data['task1_pic'].append(exg_data[:, int(TASK_1_PIC_START[i]*sps):int(TASK_1_PIC_START[i+1]*sps)])
	task_data['task1_pic'].append(exg_data[:, int(TASK_1_PIC_START[-1]*sps):int(TASK_1_END*sps)])

	task_data['task2_pic'] = []
	for i in range(len(TASK_2_PIC_START)-1):
		task_data['task2_pic'].append(exg_data[:, int(TASK_2_PIC_START[i]*sps):int(TASK_2_PIC_START[i+1]*sps)])
	task_data['task2_pic'].append(exg_data[:, int(TASK_2_PIC_START[-1]*sps):int(TASK_2_END*sps)])

	task_data['task3_pic'] = []
	for i in range(len(TASK_3_PIC_START)-1):
		task_data['task3_pic'].append(exg_data[:, int(TASK_3_PIC_START[i]*sps):int(TASK_3_PIC_START[i+1]*sps)])
	task_data['task3_pic'].append(exg_data[:, int(TASK_3_PIC_START[-1]*sps):int(TASK_3_END*sps)])

	task_data['task4_pic'] = []
	for i in range(len(TASK_4_PIC_START)-1):
		task_data['task4_pic'].append(exg_data[:, int(TASK_4_PIC_START[i]*sps):int(TASK_4_PIC_START[i+1]*sps)])
	task_data['task4_pic'].append(exg_data[:, int(TASK_4_PIC_START[-1]*sps):int(TASK_4_END*sps)])

	task_data['task5_pic'] = []
	for i in range(len(TASK_5_PIC_START)-1):
		task_data['task5_pic'].append(exg_data[:, int(TASK_5_PIC_START[i]*sps):int(TASK_5_PIC_START[i+1]*sps)])
	task_data['task5_pic'].append(exg_data[:, int(TASK_5_PIC_START[-1]*sps):int(TASK_5_END*sps)])

	task_data['task6_pic'] = []
	for i in range(len(TASK_6_PIC_START)-1):
		task_data['task6_pic'].append(exg_data[:, int(TASK_6_PIC_START[i]*sps):int(TASK_6_PIC_START[i+1]*sps)])
	task_data['task6_pic'].append(exg_data[:, int(TASK_6_PIC_START[-1]*sps):int(TASK_6_END*sps)])

	task_data['task7_pic'] = []
	for i in range(len(TASK_7_PIC_START)-1):
		task_data['task7_pic'].append(exg_data[:, int(TASK_7_PIC_START[i]*sps):int(TASK_7_PIC_START[i+1]*sps)])
	task_data['task7_pic'].append(exg_data[:, int(TASK_7_PIC_START[-1]*sps):int(TASK_7_END*sps)])

	task_data['task8_pic'] = []
	for i in range(len(TASK_8_PIC_START)-1):
		task_data['task8_pic'].append(exg_data[:, int(TASK_8_PIC_START[i]*sps):int(TASK_8_PIC_START[i+1]*sps)])
	task_data['task8_pic'].append(exg_data[:, int(TASK_8_PIC_START[-1]*sps):int(TASK_8_END*sps)])

	task_data['task9_pic'] = []
	for i in range(len(TASK_9_PIC_START)-1):
		task_data['task9_pic'].append(exg_data[:, int(TASK_9_PIC_START[i]*sps):int(TASK_9_PIC_START[i+1]*sps)])
	task_data['task9_pic'].append(exg_data[:, int(TASK_9_PIC_START[-1]*sps):int(TASK_9_END*sps)])

	if timestamp is not None:
		task_data['task1_timestamp'] = timestamp[int(TASK_1_START*sps):int(TASK_1_END*sps)]
		task_data['task2_timestamp'] = timestamp[int(TASK_2_START*sps):int(TASK_2_END*sps)]
		task_data['task3_timestamp'] = timestamp[int(TASK_3_START*sps):int(TASK_3_END*sps)]
		task_data['task4_timestamp'] = timestamp[int(TASK_4_START*sps):int(TASK_4_END*sps)]
		task_data['task5_timestamp'] = timestamp[int(TASK_5_START*sps):int(TASK_5_END*sps)]
		task_data['task6_timestamp'] = timestamp[int(TASK_6_START*sps):int(TASK_6_END*sps)]
		task_data['task7_timestamp'] = timestamp[int(TASK_7_START*sps):int(TASK_7_END*sps)]
		task_data['task8_timestamp'] = timestamp[int(TASK_8_START*sps):int(TASK_8_END*sps)]
		task_data['task9_timestamp'] = timestamp[int(TASK_9_START*sps):int(TASK_9_END*sps)]

	return task_data



def plot_head_with_electrodes(weight):
	"""
	Weight input order: 
	[Right ear 1, right ear 2, right ear 3, Af8 (side forehead at right), 
	Fp2 (near to center forehead at right), right ear 6, right ear 7, right ear 8,
	left ear 1, left ear 2, left ear 3, Af7 (side forehead at left),
	Fp1 (near to center forehead at left), left ear 6, left ear 7, left ear 8]

	https://eeglab.org/tutorials/ConceptsGuide/ICA_background.html

	"""
	# Check if the input is valid
	if len(weight) != 16:
		raise ValueError("Input must be a 1x16 vector representing electrode strengths.")

	# The mapping from input weight to electrode locations
	myorder = [15, 14, 13, 8, 9, 10, 7, 6, 5, 0, 1, 2, 3, 4, 12, 11]
	weight = np.array([weight[i] for i in myorder])

	# Define the position of virtual electrodes on the scalp to fill the figure 
	x_positions_vir = np.linspace(-0.6, 0.6, 5)
	y_positions_vir = np.linspace(-0.6, 0.6, 5)
	x_positions_vir, y_positions_vir = np.meshgrid(x_positions_vir, y_positions_vir)
	x_positions_vir = x_positions_vir.flatten()
	y_positions_vir = y_positions_vir.flatten()
	# Remove the 4 cornor points
	x_positions_vir = np.delete(x_positions_vir, [0, 4, 20, 24])
	y_positions_vir = np.delete(y_positions_vir, [0, 4, 20, 24])
	strengths_vir = np.zeros(x_positions_vir.shape)

	# Define the positions of the electrodes on the scalp
	theta = np.linspace(7/6*np.pi, 11/6*np.pi, 5) 
	radius = 1.0  # Radius of the head circle
	x_positions_vir_back = radius * np.cos(theta)
	y_positions_vir_back = radius * np.sin(theta)

	# Combine the real and virtual electrodes
	x_positions_vir = np.concatenate([x_positions_vir_back, x_positions_vir])
	y_positions_vir = np.concatenate([y_positions_vir_back, y_positions_vir])
	strengths_vir = np.concatenate([np.zeros(y_positions_vir_back.shape), strengths_vir])

	### Define the electrode of our electrodes
	x_positions_left = np.array([-0.95, -0.98, -0.95, -0.8, -0.8, -0.8])
	y_positions_left = np.array([0.15, 0, -0.15, 0.2, 0, -0.2])
	x_positions_right = np.array([0.95, 0.98, 0.95, 0.8, 0.8, 0.8])
	y_positions_right = np.array([0.15, 0, -0.15, 0.2, 0, -0.2])

	theta = np.array([np.pi/4, 2*np.pi/5, 3*np.pi/5, 3*np.pi/4])
	radius = 0.9 # Radius of the head circle
	x_positions_forehead = radius * np.cos(theta)
	y_positions_forehead = radius * np.sin(theta)

	x_positions_real = np.concatenate([x_positions_left, x_positions_right, x_positions_forehead])
	y_positions_real = np.concatenate([y_positions_left, y_positions_right, y_positions_forehead])
	strengths_real = weight  # TODO get value

	theta = np.array([np.pi/7, np.pi/2, 6*np.pi/7, np.pi/13, np.pi*12/13, np.pi*14/13, np.pi*25/13, np.pi, np.pi*2, np.pi/4, 2*np.pi/5, 3*np.pi/5, 3*np.pi/4])
	radius = 1.0  # Radius of the head circle
	x_positions_fill = radius * np.cos(theta)
	y_positions_fill = radius * np.sin(theta)
	x_positions, y_positions, strengths = x_positions_real, y_positions_real, strengths_real
	for i in range(len(x_positions_fill)):
		x_pos_fill_dist = (x_positions_real - x_positions_fill[i])**2
		y_pos_fill_dist = (y_positions_real - y_positions_fill[i])**2
		dist = np.sqrt(x_pos_fill_dist + y_pos_fill_dist)
		nearest_indices = np.argsort(dist)[:1]
		nearest_values = np.mean(strengths[nearest_indices])
		x_positions = np.concatenate([x_positions, [x_positions_fill[i]]])
		y_positions = np.concatenate([y_positions, [y_positions_fill[i]]])
		strengths = np.concatenate([strengths, [nearest_values]])
	####

	# strengths = np.concatenate([np.random.rand(len(x_positions)), strengths_vir])
	strengths = np.concatenate([strengths, strengths_vir])
	x_positions = np.concatenate([x_positions, x_positions_vir])
	y_positions = np.concatenate([y_positions, y_positions_vir])

	# Create a grid for interpolation
	grid_x, grid_y = np.mgrid[-1.5:1.5:1000j, -1.5:1.5:1000j]

	# Interpolate the strengths onto the grid
	grid_z = griddata((x_positions, y_positions), strengths, (grid_x, grid_y), method='cubic')

	# Create a figure and axis
	ax = plt.gca()

	# Draw the nose (triangle)
	nose = Polygon([[0, 1.1], [-0.3, 0.8], [0.3, 0.8]], color='w', ec='black', lw=2)
	ax.add_patch(nose)

	# Draw the ear (poly)
	ear = Polygon([[-0.8, 0], [-1, 0.2], [-1.02, 0.2],[-1.04, 0.15],[-1.04, 0],[-1.06, -0.2],[-1.04, -0.24],[-1, -0.24]], color='w', ec='black', lw=2)
	ax.add_patch(ear)
	ear = Polygon([[0.8, 0], [1, 0.2], [1.02, 0.2],[1.04, 0.15],[1.04, 0],[1.06, -0.2],[1.04, -0.24],[1, -0.24]], color='w', ec='black', lw=2)
	ax.add_patch(ear)

	# Draw the head (circle)
	head_circle = Circle((0, 0), radius, color='lightgray', ec='black', lw=2)
	ax.add_patch(head_circle)

	# Plot the heatmap using pcolormesh
	heatmap = ax.pcolormesh(grid_x, grid_y, grid_z, shading='auto', norm=mcolors.CenteredNorm(0),cmap='coolwarm')

	# Add a colorbar
	cbar = plt.colorbar(heatmap, ax=ax)
	cbar.set_label('Electrode Strength')

	# Plot the location of electrodes
	ax.scatter(x_positions_real, y_positions_real, c='black', s=10, edgecolor='black')
	
	# Set limits and aspect
	ax.set_xlim(-1.1, 1.1)
	ax.set_ylim(-1.1, 1.1)
	ax.set_aspect('equal')
	ax.axis('off')  # Turn off the axis

	# # Show the plot
	# plt.title('Electrode Strengths on Scalp')
	# plt.show()

def plot_head_IC_channels(weight):
	"""
	weights: IC * channels
	"""
	# norm weight to -1~1
	# TODO: debugging : physical meaning? work with sobi?
	# weight = (weight - np.min(weight))/(np.max(weight) - np.min(weight))*2-1

	plt.figure(figsize=(12, 12))
	for i in range(4):
		for j in range(4):
			plt.subplot(4, 4, i*4+j+1)
			plot_head_with_electrodes(weight[i*4+j])
			plt.title(f'IC {i*4+j+1}')
	# plt.show()


def channel_remapping(data, chan_ord=0):
	"""
	data: channel*sample
	Weight input order: 
	[Right ear 1, right ear 2, right ear 3, Af8 (side forehead at right), 
	Fp2 (near to center forehead at right), right ear 6, right ear 7, right ear 8,
	left ear 1, left ear 2, left ear 3, Af7 (side forehead at left),
	Fp1 (near to center forehead at left), left ear 6, left ear 7, left ear 8]
	"""
	if chan_ord == 0:
		mapping = [0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
	else:
		mapping = [0, 1, 2, 11, 12, 5, 6, 7, 8, 9, 10, 4, 3, 13, 14, 15]
	data = np.array([data[i,:] for i in mapping])
	return data

def calculate_spectrum(data_len, sps, lo, hi):
	"""
	data: channel*sample

	calculate the fft idx of the lo Hz and hi Hz
	"""
	# fft_freq = np.fft.fftfreq(data_len, 1/sps)
	fft_freq = np.fft.rfftfreq(data_len, 1/sps)
	lo_idx = np.argmin(np.abs(fft_freq - lo))
	hi_idx = np.argmin(np.abs(fft_freq - hi))
	return lo_idx, hi_idx

def calculate_metric(data, sps):
	"""
	data: channel*sample
	"""
	# delta: 1-4
	lo, hi = 1, 4
	lo_idx, hi_idx = calculate_spectrum(data.shape[1], sps, lo, hi)
	# delta_spec = np.sum(np.abs(np.fft.fft(data, axis=1)[:,lo_idx:hi_idx])**2, axis=1)
	delta_spec = np.sum(np.abs(np.fft.rfft(data, axis=1)[:,lo_idx:hi_idx])**2, axis=1)
	# delta_spec = np.mean(delta_spec, axis=0)

	# theta: 4-8
	lo, hi = 4, 8
	lo_idx, hi_idx = calculate_spectrum(data.shape[1], sps, lo, hi)
	# theta_spec = np.sum(np.abs(np.fft.fft(data, axis=1)[:,lo_idx:hi_idx])**2, axis=1)
	theta_spec = np.sum(np.abs(np.fft.rfft(data, axis=1)[:,lo_idx:hi_idx])**2, axis=1)
	# theta_spec = np.mean(theta_spec, axis=0)

	# alpha: 8-12
	lo, hi = 8, 13
	lo_idx, hi_idx = calculate_spectrum(data.shape[1], sps, lo, hi)
	# alpha_spec = np.sum(np.abs(np.fft.fft(data, axis=1)[:,lo_idx:hi_idx])**2, axis=1)
	alpha_spec = np.sum(np.abs(np.fft.rfft(data, axis=1)[:,lo_idx:hi_idx])**2, axis=1)
	# alpha_spec = np.mean(alpha_spec, axis=0)

	# beta: 13-30
	lo, hi = 13, 30
	lo_idx, hi_idx = calculate_spectrum(data.shape[1], sps, lo, hi)
	# beta_spec = np.sum(np.abs(np.fft.fft(data, axis=1)[:,lo_idx:hi_idx])**2, axis=1)
	beta_spec = np.sum(np.abs(np.fft.rfft(data, axis=1)[:,lo_idx:hi_idx])**2, axis=1)
	# beta_spec = np.mean(beta_spec, axis=0)

	# gamma: 31-40
	lo, hi = 30, 40
	lo_idx, hi_idx = calculate_spectrum(data.shape[1], sps, lo, hi)
	# gamma_spec = np.sum(np.abs(np.fft.fft(data, axis=1)[:,lo_idx:hi_idx])**2, axis=1)
	gamma_spec = np.sum(np.abs(np.fft.rfft(data, axis=1)[:,lo_idx:hi_idx])**2, axis=1)
	# gamma_spec = np.mean(gamma_spec, axis=0)

	# whole 0.3-60
	lo, hi = 0.3, 60
	lo_idx, hi_idx = calculate_spectrum(data.shape[1], sps, lo, hi)
	# whole_spec = np.sum(np.abs(np.fft.fft(data, axis=1)[:,lo_idx:hi_idx])**2, axis=1)
	whole_spec = np.sum(np.abs(np.fft.rfft(data, axis=1)[:,lo_idx:hi_idx])**2, axis=1)
	# whole_spec = np.mean(whole_spec, axis=0)

	return delta_spec, theta_spec, alpha_spec, beta_spec, gamma_spec, whole_spec

def calculate_filted(data, sps):
	"""
	data: channel*sample
	"""
	delta_ft = butter(10, [0.3, 4], 'bp', fs=sps, output='sos')
	delta_sig = sosfiltfilt(delta_ft, data, axis=0)
	delta_spec = np.mean(np.abs(np.fft.fft(delta_sig, axis=0))**2, axis=0)

	# theta: 4-8
	theta_ft = butter(10, [4, 7], 'bp', fs=sps, output='sos')
	theta_sig = sosfiltfilt(theta_ft, data, axis=0)
	theta_spec = np.mean(np.abs(np.fft.fft(theta_sig, axis=0))**2, axis=0)

	# alpha: 8-12
	alpha_ft = butter(10, [8, 13], 'bp', fs=sps, output='sos')
	alpha_sig = sosfiltfilt(alpha_ft, data, axis=0)
	alpha_spec = np.mean(np.abs(np.fft.fft(alpha_sig, axis=0))**2, axis=0)

	# beta: 13-30
	beta_ft = butter(10, [13, 30], 'bp', fs=sps, output='sos')
	beta_sig = sosfiltfilt(beta_ft, data, axis=0)
	beta_spec = np.mean(np.abs(np.fft.fft(beta_sig, axis=0))**2, axis=0)

	# gamma: 31-40
	gamma_ft = butter(10, [30, 40], 'bp', fs=sps, output='sos')
	gamma_sig = sosfiltfilt(gamma_ft, data, axis=0)
	gamma_spec = np.mean(np.abs(np.fft.fft(gamma_sig, axis=0))**2, axis=0)


def extract_trial_list_by_json(json_file, trial_dict=None):
	"""
	input: json_file: the json file where the data is stored
	return trial_list = [[date_time, subject, task_type, task_id, pic_id], ...]
	"""
	print(f'use json file: {json_file} in [extract_trial_list_by_json]')
	if trial_dict is None:
		f = open(json_file)
		trials_dict = json.load(f)
	trial_list = []
	for k, v in trials_dict.items():
		file_name = v['file']
		if 'min' in json_file:
			subject, date_time, task_type, task_id, pic_id = file_name.split('-')
		else:
			subject, date_time, task_type, task_id, pic_id = file_name.split('-')[:-2]
		trial_list.append(f'{subject}/{date_time}/{task_type}/{task_id}/{pic_id}')
	trial_list = set(trial_list)
	trial_set = []
	for t in trial_list:
		trial_set.append(t.split('/'))
	return trial_set

def split_records_by_subjects(records, train_subjects=None, valid_subjects=None, test_subjects=None):
	# [[subject, date_time, task_type, task_id, pic_id], ...]
	train_records = []
	test_records = []
	valid_records = []
	for record in records:
		if record[0] in test_subjects:
			test_records.append(record)
		if valid_subjects != None and record[0] in valid_subjects:
			valid_records.append(record)
		elif train_subjects is None and record[0] not in test_subjects: #and valid_subjects != None and record[0] not in valid_subjects:
			train_records.append(record)
		elif train_subjects is not None and record[0] in train_subjects:
			train_records.append(record)
	if valid_subjects == None:
		valid_records = None
	return train_records, valid_records, test_records

def split_records_single_subject(records, ratios=[0.8, 0, 0.2], seed=43):
	records = sorted(records, key=lambda x: x[1])
	current_user = records[0][1]
	user_records = []
	train_records, valid_records, test_records = [], [], []
	for i, re in enumerate(records):
		user_records.append(re)
		if re[1] != current_user:
			current_user = re[1]
			train_records_user, valid_records_user, test_records_user = split_records_by_random(user_records[:-1], ratios, seed)
			train_records.extend(train_records_user)
			if valid_records_user is not None:
				valid_records.extend(valid_records_user) 
			test_records.extend(test_records_user)
			assert len(train_records_user) + len(test_records_user) == len(user_records[:-1]), "length is not correct, {}".format(current_user)
			user_records = [user_records[-1]]
		if i == len(records) - 1:
			# last record
			train_records_user, valid_records_user, test_records_user = split_records_by_random(user_records, ratios, seed)
			train_records.extend(train_records_user)
			if valid_records_user is not None:
				valid_records.extend(valid_records_user) 
			test_records.extend(test_records_user)
			assert len(train_records_user) + len(test_records_user) == len(user_records), "length is not correct, {}".format(current_user)
	if ratios[1] == 0:
		valid_records = None
	return train_records, valid_records, test_records

def split_records_by_random(records, ratios=[0.8, 0, 0.2], seed=43):
	np.random.seed(seed)
	# records_idx = np.linspace(0, len(records)-1, len(records)).astype(int)
	# np.random.shuffle(records_idx)
	records = sorted(records)
	records_idx = np.random.permutation(len(records))
	# random.shuffle(records)
	records = [records[i] for i in records_idx]
	n_total = len(records)
	valid_offset, test_offset = [int(n_total * r) for r in ratios[1:]]
	total_offset = valid_offset + test_offset
	train_records = records[:n_total-total_offset]
	if ratios[1] > 0:
		valid_records = records[n_total-total_offset : n_total-test_offset]
	else:
		valid_records = None
	test_records = records[n_total-test_offset:]
	return train_records, valid_records, test_records
def split_test_records_by_ratio(test_records, target_ratio=0.2, seed=42):
	"""
	Splits test records by selecting a ratio of records from each subject.
	
	Args:
		test_records: List of [date_time, subject, task_type, task_id, pic_id]
		target_ratio: Float between 0 and 1, ratio of records to select per subject
		seed: Random seed for reproducibility
	
	Returns:
		target_records: Selected records
		remaining_records: Remaining records
	"""
	np.random.seed(seed)
	
	# Group records by subject
	subject_records = {}
	for record in test_records:
		subject = record[1]
		if subject not in subject_records:
			subject_records[subject] = []
		subject_records[subject].append(record)
	
	target_records = []
	
	# Select ratio of records for each subject
	for subject, records in subject_records.items():
		n_select = max(1, int(len(records) * target_ratio))
		indices = np.random.permutation(len(records))
		
		# Split into selected and remaining
		selected = [records[i] for i in indices[:n_select]]

		target_records.extend(selected)
	
	return target_records

def filter_records_by_cog_task(records, cog_task):
	"""
	Filter records by cog_task
	"""
	if records is None:
		return None
	if cog_task == 'all':
		return records
	filtered_records = []
	for record in records:
		if record[2] == cog_task:
			filtered_records.append(record)
	return filtered_records


def get_records(cfg):
	records = extract_trial_list_by_json(cfg['sliced_trials_json'])
	if cfg['test_subject'] is not None:
		### split the records by the subjects, leave-subject-out
		train_records, val_records, test_records = split_records_by_subjects(
			records, train_subjects=cfg['train_subject'], valid_subjects=cfg['valid_subject'],
			test_subjects=cfg['test_subject'])
	else:
		### randomly split the records from the same subjects
		train_records, val_records, test_records = split_records_single_subject(
			records, seed=cfg['seed'])

	train_records = filter_records_by_cog_task(train_records, cfg['cog_task'])
	val_records = filter_records_by_cog_task(val_records, cfg['cog_task'])
	test_records = filter_records_by_cog_task(test_records, cfg['cog_task'])

	print("train records: ", train_records)
	print("val records: ", val_records)
	print("test records: ", test_records)

	### TODO: adjust for debug
	# train_records = train_records[:2]
	# test_records = test_records[:2]

	subj = []
	for t in test_records:
		ts = t[0]
		subj.append(ts)
	subj = sorted(list(set(subj)))
	logger.info(f'test_subjects:{subj}')

	if cfg['is_split_trials_json_dict']:
		record_indices = get_record_trials_dict(cfg['sliced_trials_json'])
		train_records = split_trials_json_by_records(train_records, record_indices)
		val_records = split_trials_json_by_records(val_records, record_indices)
		test_records = split_trials_json_by_records(test_records, record_indices)

	return train_records, val_records, test_records

def find_json_indices_by_record(record, trial_dict):
	indices = []
	for k, v in trial_dict.items():
		key = f'{record[0]}-{record[1]}-{record[2]}-{record[3]}-{record[4]}'
		if key in v['file']:
			indices.append(int(k))
	return indices

# def split_trials_json_by_records(records, json_file):
#### original slow version
# 	# [[subject, date_time, task_type, task_id, pic_id], ...] = records
# 	f = open(json_file)
# 	trials_dict = json.load(f)
# 	trials_indices = []
# 	if records is None:
# 		return None
# 	for record in records:
# 		trials_indices.extend(find_json_indices_by_record(record, trials_dict))
# 	return np.array(trials_indices)

def get_record_trials_dict(json_file):
	with open(json_file) as f:
		data_dict = json.load(f)
	record_indices = {}
	for k, v in data_dict.items():
		key = f"{v['subject']}-{v['date_time']}-{v['data_type']}-{v['task_no']}-{v['pic_no']}"
		if key not in record_indices:
			record_indices[key] = []
		record_indices[key].append(int(k))
	return record_indices
			

def split_trials_json_by_records(records, record_indices):
	"""
	Split trials based on records and return their indices from the JSON file.

	Parameters
	----------
	records : list
		A list of records where each record is in the format 
		[date_time, subject, task_type, task_id, pic_id].
	record_indices : dict
		A dictionary mapping record keys to their indices.

	Returns
	-------
	np.array
		An array of indices corresponding to the records found in the JSON.
	"""

	trials_indices = []
	if records is None or len(records) == 0:
		return None
	for record in records:
		record_key = f"{record[0]}-{record[1]}-{record[2]}-{record[3]}-{record[4]}"
		if record_key in record_indices.keys():
			trials_indices.extend(record_indices[record_key])  # Extend the list with all indices

	return np.array(trials_indices)

def split_data_json_by_subject_activity(data_json, subject, data_type, task_no, pic_no=None):
	with open(data_json, "r") as f:
		data_info = json.load(f)
	selected_items_indices = []
	for ii in data_info.keys():
		item = data_info[ii]
		if item["subject"] == subject and item["data_type"] == data_type and item["task_no"] == task_no:
			if pic_no is None:
				selected_items_indices.append(ii)
			else:
				if item["pic_no"] == pic_no:
					selected_items_indices.append(ii)
	return selected_items_indices

def get_unique_subjects(records):
	"""
	Extract unique subjects from the list of records.

	Parameters
	----------
	records : list
		A list of records where each record is in the format 
		[date_time, subject, task_type, task_id, pic_id].

	Returns
	-------
	unique_subjects : set
		A set of unique subjects.
	"""
	unique_subjects = {record[1] for record in records}  # Using a set comprehension to collect unique subjects
	return unique_subjects

def get_subject_activity_dict(json_file, indices=None):
	# [[date_time, subject, task_type, task_id, pic_id], ...]
	if indices is None:
		records = extract_trial_list_by_json(json_file)
	else:
		data_dict = json.load(open(json_file))
		selected_items = {}
		for idx in indices:
			selected_items[str(idx)] = data_dict[str(idx)]
		records = extract_trial_list_by_json(json_file, selected_items)
	subjects = get_unique_subjects(records)
	subject_dict = {}
	for subject in subjects:
		subject_dict[subject] = []
	for rec in records:
		subject_dict[rec[1]].append(f'{rec[0]}-{rec[2]}-{rec[3]}-{rec[4]}')
	return subject_dict

def parse_model_save_path(log_file):
	model_save_path = None
	with open(log_file, 'r') as file:
		for line in file:
			if 'Model will be saved at' in line:
				# Extract the path from the line
				model_save_path = line.split('Model will be saved at: [')[1].split(']')[0].strip()
				break  # Exit the loop after finding the first occurrence
	return model_save_path

def set_model_dict(user_list_ids, model_save_path):
	model_dict = {}
	for idx in user_list_ids:
		model_dict[str(idx)] = model_save_path
	return model_dict

def str_to_bool(value):
	"""Convert a string to a boolean value."""
	if value.lower() in ('true', '1', 'yes'):
		return True
	elif value.lower() in ('false', '0', 'no'):
		return False
	else:
		raise argparse.ArgumentTypeError(f"Boolean value expected for {value}")


def transpose_nested_list(nested_list):
		"""
		Transpose a nested list structure.
		
		Input: [[m1_view1, m1_view2], [m2_view1, m2_view2]]
		Output: [[view1_m1, view1_m2], [view2_m1, view2_m2]]
		"""
		# Get the number of modalities and views
		num_modalities = len(nested_list)
		num_views = len(nested_list[0])
		
		# Create the transposed structure
		transposed = [[] for _ in range(num_views)]
		
		# Fill the transposed structure
		for mod_idx in range(num_modalities):
			for view_idx in range(num_views):
				transposed[view_idx].append(nested_list[mod_idx][view_idx])
		
		return transposed

def transpose_aug_list(aug_list):
	"""
	Transpose a list of augmented data.
	
	Input: [[view1_m1, view1_m2], [view2_m1, view2_m2]]
	Output: [[m1_view1, m1_view2](stacked), [m2_view1, m2_view2](stacked)]
	"""
	if not isinstance(aug_list[0], list):
		return aug_list
	# Get the number of modalities and views
	num_modalities = len(aug_list[0])
	num_views = len(aug_list)
	
	# Create the transposed structure
	transposed = []
		
	# Fill the transposed structure
	for mod_idx in range(num_modalities):
		transposed.append(torch.stack([aug_list[view_idx][mod_idx] for view_idx in range(num_views)], axis=0))

	return transposed


def select_data_set(data_type):
	if data_type == 'clean_0426':
		return balanced_subject_split.Score_Balanced_Subjects_Clean()
	elif data_type == 'all_0426':
		return balanced_subject_split.Score_Balanced_Subjects_All()
	else:
		raise ValueError(f"Invalid data type: {data_type}")


def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
	mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
	mask = mask.bernoulli_(mask_ratio)
	return mask

def check_shapes(features_list):
	if not features_list:
		return True
	if isinstance(features_list[0], list):
		first_length = len(features_list[0])
		for features in features_list:
			if len(features) != first_length:
				return False
		return True
	if isinstance(features_list[0], np.ndarray):
		first_shape = features_list[0].shape
		for i, features in enumerate(features_list):
			if features.shape != first_shape:
				return False
		return True
	return True

def aggregate_features_within_task(task_features, method='mean'):
	if method == 'mean':
		if isinstance(task_features, torch.Tensor):
			return torch.mean(task_features, dim=0, keepdim=True)
		else:
			return np.mean(task_features, axis=0, keepdims=True)
	elif method == 'none':
		return task_features
	elif 'random_sample' in method:
		sample_ratio = float(method.split('_')[-1])
		if isinstance(task_features, torch.Tensor):
			num_samples = max(1, int(task_features.shape[0] * sample_ratio))
			random_indices = torch.randperm(task_features.shape[0])[:num_samples]
			if num_samples > 1:
				return torch.mean(task_features[random_indices, ...], dim=0, keepdim=True)
			else:
				return task_features[random_indices, ...]
		else:
			num_samples = max(1, int(task_features.shape[0] * sample_ratio))
			random_indices = np.random.choice(task_features.shape[0], num_samples, replace=False)
			if num_samples > 1:
				return np.mean(task_features[random_indices, ...], axis=0, keepdims=True)
			else:
				return task_features[random_indices, ...]
	elif 'split' in method:
		sample_ratio = float(method.split('-')[-1])
		#  n_slice, n_dim
		if task_features.shape[0] > 30:
			n_pics = 1
		elif task_features.shape[0] == 18:
			n_pics = 6
		else:
			n_pics = 10
		n_slice_per_pic = max(1, task_features.shape[0] // n_pics)
		num_samples = max(1, int(n_slice_per_pic * sample_ratio))

		if isinstance(task_features, torch.Tensor):
			new_ret = torch.zeros((n_pics, task_features.shape[1]), device=task_features.device)
			for i in range(n_pics):
				random_indices = torch.randperm(n_slice_per_pic)[:num_samples] + i * n_slice_per_pic
				if num_samples > 1:
					new_ret[i, ...] = torch.mean(task_features[random_indices, ...], dim=0, keepdim=False)
				else:
					new_ret[i, ...] = task_features[random_indices, ...]
		else:
			new_ret = np.zeros((n_pics, task_features.shape[1]))
			for i in range(n_pics):
				random_indices = np.random.choice(n_slice_per_pic, num_samples, replace=False) + i * n_slice_per_pic
				if num_samples > 1:
					new_ret[i, ...] = np.mean(task_features[random_indices, ...], axis=0, keepdims=False)
				else:
					new_ret[i, ...] = task_features[random_indices, ...]
		return new_ret
	else:
		raise ValueError(f"Invalid aggregation method: {method}")

def aggregate_features_across_tasks(features_list, method='concat'):
	if not check_shapes(features_list):
		### the tasks have different elements, so we cannot aggregate them
		return features_list
	
	# Handle both numpy arrays and torch tensors
	if isinstance(features_list[0], torch.Tensor):
		# Tensor version
		if method == 'concat':
			features = torch.cat(features_list, dim=0)
			# Reshape to (1, -1) to match the original behavior
			features = features.view(1, -1)
		elif method == 'mean':
			features = torch.stack(features_list, dim=0)
			features = torch.mean(features, dim=0, keepdim=True)
		elif method == 'none':
			features = features_list
		else:
			raise ValueError(f"Invalid aggregation method: {method}")
	else:
		# Numpy version
		features_list = np.array(features_list)
		if method == 'concat':
			features = np.concatenate(features_list, axis=0)
			features = features.reshape(1, -1)
		elif method == 'mean':
			features = np.mean(features_list, axis=0, keepdims=True)
		elif method == 'none':
			features = features_list
		else:
			raise ValueError(f"Invalid aggregation method: {method}")
	
	return features

def intra_task_sample_indx_selection(task_ExG_indices, method='random_sample'):
	### task_ExG_indices: [num_samples]
	if 'random_sample' in method:
		sample_ratio = float(method.split('-')[-1])
		num_samples = max(1, int(task_ExG_indices.shape[0] * sample_ratio))
		random_indices = np.random.choice(task_ExG_indices.shape[0], num_samples, replace=False)
		exg_indices = task_ExG_indices[random_indices, ...]
	elif method == 'mean' or method == 'all' or method == 'none':
		exg_indices = task_ExG_indices
	else:
		raise ValueError(f"Invalid aggregation method: {method}")
	return exg_indices
	

def get_data_dict_list_idx(data_dict, list_idx):
	return {k: v[list_idx] for k, v in data_dict.items()}

def get_data_dict_list_to_item(data_dict):
	new_data_dict = {}
	is_convert = False
	for k in data_dict.keys():
		new_data_dict[k] = []
		v = data_dict[k]
		if isinstance(v, list):
			if isinstance(v[0], list):
				is_convert = True
				for ii in v:
					new_data_dict[k].append(ii[0])
	if is_convert:
		return new_data_dict
	else:
		return data_dict

def convert_data_dict_to_list(data_dict):
	"""Convert data dict to list"""
	keys = list(data_dict.keys())
	data_list = []
	for i in range(len(keys)):
		data_list.append(data_dict[keys[i]])
	return data_list

def to_device(data, device):
	"""Move data to device, handling nested structures"""
	if isinstance(data, dict):
		return {k: to_device(v, device) for k, v in data.items()}
	elif isinstance(data, (list, tuple)):
		return [to_device(x, device) for x in data]
	elif hasattr(data, 'to'):  # Check if it's a tensor-like object with 'to' method
		return data.to(device)
	elif isinstance(data, np.ndarray):
		return torch.from_numpy(data).to(device)
	else:
		return data  # Return as is if it's not a tensor or container
	

def select_trials_json(cfg, data_type):
	if 'sliced_data_folder_clean' in cfg.keys() and 'sliced_data_folder_all' in cfg.keys():
		if 'clean' in data_type:
			folder = cfg['sliced_data_folder_clean']
			trials_json = cfg['sliced_trials_json_clean']
		else:
			folder = cfg['sliced_data_folder_all']
			trials_json = cfg['sliced_trials_json_all']
	return folder, trials_json

def test_gpu(gpu, memory=20):
	pynvml.nvmlInit()
	handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
	info = pynvml.nvmlDeviceGetMemoryInfo(handle)
	if info.free / 1024 / 1024 / 1024 < memory:
		return False
	else:
		return True

def scan_available_gpus(memory=20, gpu_scope=[0, 1, 2, 3, 4, 5, 6, 7]):
	pynvml.nvmlInit()
	if gpu_scope is None:
		num_gpus = pynvml.nvmlDeviceGetCount()
	else:
		num_gpus = len(gpu_scope)

	available_gpus = []
	flag = False
	for i in range(num_gpus):
		if gpu_scope is not None:
			gpu_idx = gpu_scope[i]
		else:
			gpu_idx = i
		if test_gpu(gpu_idx, memory):
			available_gpus.append(gpu_idx)
			flag = True
	if flag:
		return available_gpus
	else:
		return None


def read_task_score(patient_info_path):
	ret = {}
	patient_info = read_xlsx_to_dict(patient_info_path)
	
	# Find the last non-None index in patient_info["id"]
	last_valid_idx = -1
	for i, item in enumerate(patient_info["id"]):
		if item is not None:
			last_valid_idx = i
	
	id = np.array(patient_info["id"])[:last_valid_idx]
	xlsx_date = patient_info["Date"][:last_valid_idx]
	xlsx_date = ['_'.join(item.split('.')) for item in xlsx_date]
	MoCA = patient_info["MoCA"]
	MMSE = patient_info["MMSE"]

	for i in range(len(MoCA)):
		if not isinstance(MoCA[i], int):
			MoCA[i] = -1
		if not isinstance(MMSE[i], int):
			MMSE[i] = -1

	task_score = []
	for i in range(7):
		task_score.append(patient_info[f"Task{i+1}"])
	
	MMSE_task_score = []
	for i in range(6):
		MMSE_task_score.append(patient_info[f"MMSE_Task{i+1}"])
	
	task_score, MMSE_task_score = np.array(task_score, dtype=float).T[:last_valid_idx,:], np.array(MMSE_task_score, dtype=float).T[:last_valid_idx,:]
	for i in range(7):
		task_score[:last_valid_idx,i] = task_score[:last_valid_idx,i]
	
	for i in range(6):
		MMSE_task_score[:last_valid_idx,i] = MMSE_task_score[:last_valid_idx,i]

	for i in range(len(id)):
		# print(id[i])
		for j in range(7):
			if task_score[i,j] < 0:
				task_score[i,j] = (MoCA[i]/30 * MOCA_TASK_SCORE_MAX[j])
		for j in range(6):
			if MMSE_task_score[i,j] < 0:
				MMSE_task_score[i,j] = (MMSE[i]/30 * MMSE_TASK_SCORE_MAX[j])

	for i in range(len(id)):
		item = {}
		item['moca_task_score'] = task_score[i]
		item['mmse_task_score'] = MMSE_task_score[i]
		item['moca'] = MoCA[i]
		item['mmse'] = MMSE[i]
		ret[f'{id[i]}-{xlsx_date[i]}'] = item
	
	return ret

def update_subject_task_indices(subject_task_indices, subject_taskscore_dict):
	for subject in subject_task_indices.keys():
		for session in subject_task_indices[subject].keys():
			MoCA, MMSE = subject_task_indices[subject][session]['cognitive_scores']['MoCA'], subject_task_indices[subject][session]['cognitive_scores']['MMSE']
			for sub_date in subject_taskscore_dict.keys():
				if subject in sub_date:
					if MoCA == subject_taskscore_dict[sub_date]['moca'] and MMSE == subject_taskscore_dict[sub_date]['mmse']:
						subject_task_indices[subject][session]['cognitive_scores']['MoCA_taskscore'] = subject_taskscore_dict[sub_date]['moca_task_score']
						subject_task_indices[subject][session]['cognitive_scores']['MMSE_taskscore'] = subject_taskscore_dict[sub_date]['mmse_task_score']
						break    

	return subject_task_indices