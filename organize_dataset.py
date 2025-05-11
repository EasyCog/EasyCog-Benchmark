import numpy as np
import os
import json
from utils import logger
from data_processing.features import *
from data_processing.excel_operation import read_xlsx_to_dict


feat_name = 'asreog_filter_order3_all_data'

folder_path = f'./sliced/{feat_name}'
saved_json_folder = f'./data_json_files/'
feature_folder_path = f'./processed_feat/{feat_name}'
patient_info_path = './Patient_Info_dataset.xlsx'


def create_or_update_dict_items(data_dict, key, value):
	"""
	Create a new key-value pair in the dictionary or update an existing key's value.
	
	Args:
		data_dict (dict): The dictionary to be modified
		key (str): The key to be added or updated
		value (dict): The value to be assigned to the key
		
	Returns:
		dict: The updated dictionary
	"""
	if key not in list(data_dict.keys()):
		data_dict[key] = value
	else:
		data_dict[key].update(value)
	return data_dict

def create_json_with_data_folder(folder_path, json_file, specific_subject=None):
	"""
	Create or update a JSON file with data from a specified folder.
	Processes each file in the folder and extracts relevant information including subject details,
	EEG/EOG features, and cognitive assessment scores.
	
	Args:
		folder_path (str): Path to the folder containing data files
		json_file (str): Path where the JSON file will be saved
		specific_subject (str, optional): If provided, only process files for this subject
	"""
	data_dict = {}
	counter = -1
	if os.path.exists(json_file) is False:
		# create a file with nothing
		with open(json_file, "w", encoding="utf-8") as f:
			f.write('')
	else:
		try:
			with open(json_file, "r", encoding="utf-8") as f:
				data_dict = json.load(f)
		except (json.JSONDecodeError, FileNotFoundError):
			data_dict = {}
	
	files = os.listdir(folder_path)
	for file in files:
		filepath = os.path.join(folder_path, file)
		if os.path.isfile(filepath):
			counter += 1
			if counter % 1000 == 0:
				logger.info(counter)

			### e.g., 070_patient-2025_01_17_09_00_43-resting-task9-pic0-8500-8875.npz
			
			subject, date_time, data_type, task_no, pic_no = file.split("-")[:5]
			eeg, eog, et, moca, mmse, moca_task_score, mmse_task_score, eeg_std_all, eog_mean_all, eeg_mean_all, eog_std_all = read_sliced_feat(filepath)

			if specific_subject is not None and specific_subject not in subject:
				continue
			data_dict = create_or_update_dict_items(data_dict, key=str(counter), value={'root': folder_path})
			data_dict[str(counter)].update({
				'root': folder_path,
				'subject': subject,
				'date_time': date_time,
				'data_type': data_type,
				'task_no': task_no,
				'pic_no': pic_no,
				'file': file,
				'MoCA': moca.item(),
				'MMSE': mmse.item(),
				'eeg_std_all': eeg_std_all.item(),
				'eog_mean_all': eog_mean_all.item(),
				'eeg_mean_all': eeg_mean_all.item(),
				'eog_std_all': eog_std_all.item(),
				'moca_task_score': moca_task_score.item(),
				'mmse_task_score': mmse_task_score.item()
			})  
			
	with open(f'{json_file}', 'w', encoding='utf-8') as f:
		json.dump(data_dict, f, indent=2)

def change_json_root(json_file, saved_json_file, new_root):
	"""
	Update the root path in a JSON file and save it to a new location.
	
	Args:
		json_file (str): Path to the source JSON file
		saved_json_file (str): Path where the modified JSON will be saved
		new_root (str): New root path to be set for all entries
	"""
	json_folder, json_name = os.path.split(json_file)
	data_dict = json.load(open(json_file, 'r'))
	new_data_dict = {}
	for k, v in data_dict.items():
		v['root'] = new_root
		new_data_dict[k] = v
	
	with open(os.path.join(json_folder, saved_json_file), 'w', encoding='utf-8') as f:
		json.dump(new_data_dict, f, indent=2)

def _add_disease_info_to_json(json_file, patient_info_path, saved_json_file=None):
	"""
	Add disease information to each entry in the JSON file based on patient records.
	Maps disease categories to numerical codes (0: Control, 1: PD, 2: AD, 3: VD/VaD).
	
	Args:
		json_file (str): Path to the source JSON file
		patient_info_path (str): Path to the Excel file containing patient information
		saved_json_file (str, optional): Path where the modified JSON will be saved
	"""
	data_dict = json.load(open(json_file, 'r'))
	patient_info = read_xlsx_to_dict(patient_info_path)
	id = np.array(patient_info["id"])[:87]
	xlsx_date = patient_info["Date"][:87]
	disease_info = patient_info["Disease"]
	
	for k, v in data_dict.items():
		subject = str(v['subject'])
		date = str(v['date_time'])
		date_str = '.'.join(date.split('_')[:3])
		subject_idx_list = np.where(id == subject)[0]
		if len(subject_idx_list) > 0:
			for s_idx in subject_idx_list:
				if date_str == xlsx_date[s_idx]:
					subject_idx = s_idx
		else:
			subject_idx = subject_idx_list[0]
		disease_str = disease_info[subject_idx]
		if 'AD' in disease_str:
			v['disease'] = 2
		elif 'Control' in disease_str or 'control' in disease_str:
			v['disease'] = 0
		elif 'VD' in disease_str or 'vd' in disease_str or 'VaD' in disease_str:
			v['disease'] = 3
		elif 'PD' in disease_str:
			v['disease'] = 1
		else:
			if subject != '002_patient':
				logger.error(f'Disease info for subject {subject} is not found or not recognized')
			v['disease'] = -1

	with open(saved_json_file, 'w', encoding='utf-8') as f:
		json.dump(data_dict, f, indent=2)


def _add_features_to_json(json_file, processed_feat_path, saved_json_file=None):
	"""
	Add feature file information to each entry in the JSON file.
	Updates entries with feature file paths and available feature names.
	
	Args:
		json_file (str): Path to the source JSON file
		processed_feat_path (str): Path to the directory containing processed feature files
		saved_json_file (str, optional): Path where the modified JSON will be saved
	"""
	if saved_json_file is None:
		saved_json_file = json_file
	data_dict = json.load(open(json_file, 'r'))
	for k, v in data_dict.items():
		file = v['file'].split('.')[0] + '-feat.npz'
		feat_file = os.path.join(processed_feat_path, file)
		if os.path.exists(feat_file):
			v['feat_root_path'] = processed_feat_path
			v['feat_file'] = feat_file
			v['features'] = FEATURE_NAMES
		else:
			logger.error(f'File {feat_file} does not exist')
	
	with open(saved_json_file, 'w', encoding='utf-8') as f:
		json.dump(data_dict, f, indent=2)

def update_moca_mmse_score(json_file, patient_info_path, saved_json_file=None):
	"""
	Update MoCA and MMSE scores in the JSON file based on patient records.
	Handles special cases and validates score updates.
	
	Args:
		json_file (str): Path to the source JSON file
		patient_info_path (str): Path to the Excel file containing patient information
		saved_json_file (str, optional): Path where the modified JSON will be saved
	"""
	if saved_json_file is None:
		saved_json_file = json_file
	data_dict = json.load(open(json_file, 'r'))
	patient_info = read_xlsx_to_dict(patient_info_path)
	id = np.array(patient_info["id"])[:90]
	xlsx_date = patient_info["Date"][:90]
	MoCA = patient_info["MoCA"]
	MMSE = patient_info["MMSE"]
	for k, v in data_dict.items():
		subject = str(v['subject'])
		date = str(v['date_time'])
		date_str = '.'.join(date.split('_')[:3])
		subject_idx_list = np.where(id == subject)[0]
		if len(subject_idx_list) > 0:
			for s_idx in subject_idx_list:
				if date_str == xlsx_date[s_idx]:
					subject_idx = s_idx
		else:
			subject_idx = subject_idx_list[0]
		subject_moca = MoCA[subject_idx]
		if not isinstance(subject_moca,int):
			subject_moca = -1

		subject_mmse = MMSE[subject_idx]
		if not isinstance(subject_mmse,int):
			subject_mmse = -1

		if subject == '070_patient' and date_str == '2025.01.16':
			if date.split('_')[3] == '12':
				subject_moca = 28
				subject_mmse = 28
			elif date.split('_')[3] == '20':
				subject_moca = 28
				subject_mmse = 29

		if v['MoCA'] != subject_moca:
			print("Updating MoCA score for subject: ", v['subject'])
			v['MoCA'] = subject_moca
		if v['MMSE'] != subject_mmse:
			print("Updating MMSE score for subject: ", v['subject'])
			v['MMSE'] = subject_mmse
	
	with open(saved_json_file, 'w', encoding='utf-8') as f:
		json.dump(data_dict, f, indent=2)

def delete_given_subject_from_json(json_file, subject_list, saved_json_file=None):
	"""
	Remove entries for specified subjects from the JSON file and reindex remaining entries.
	
	Args:
		json_file (str): Path to the source JSON file
		subject_list (list): List of subject identifiers to be removed
		saved_json_file (str, optional): Path where the modified JSON will be saved
	"""
	data_dict = json.load(open(json_file, 'r'))
	delete_idx = []
	for k, v in data_dict.items():
		if v['subject'] in subject_list:
			delete_idx.append(k)
	for idx in delete_idx:
		del data_dict[idx]

	counter = -1
	new_data_dict = {}
	for k, v in data_dict.items():
		counter += 1
		new_data_dict[str(counter)] = v
	
	with open(saved_json_file, 'w', encoding='utf-8') as f:
		json.dump(new_data_dict, f, indent=2)

def check_key_order(json_file):
	"""
	Verify that the keys in the JSON file are in sequential order.
	Prints a message if any key is out of order.
	
	Args:
		json_file (str): Path to the JSON file to be checked
	"""
	data_dict = json.load(open(json_file, 'r'))
	counter = -1
	for k, v in data_dict.items():
		counter += 1
		if int(k) != counter:
			print(f'{k} is not in order')
			break
	print(f'{counter} is the last key')


def combine_json_files(json1, json2, saved_json_file):
	"""
	Merge two JSON files into a single file, maintaining sequential key ordering.
	
	Args:
		json1 (str): Path to the first JSON file
		json2 (str): Path to the second JSON file
		saved_json_file (str): Path where the combined JSON will be saved
	"""
	data_dict1 = json.load(open(json1, 'r'))
	data_dict2 = json.load(open(json2, 'r'))
	max_k = max(int(k) for k in data_dict1.keys())
	counter = max_k
	for k, v in data_dict2.items():
		counter += 1
		data_dict1 = create_or_update_dict_items(data_dict1, key=str(counter), value=v)
	with open(saved_json_file, 'w', encoding='utf-8') as f:
		json.dump(data_dict1, f, indent=2)


def extract_clean_data_from_json(json_file, saved_json_file, clean_data_subjects):
	"""
	Extract entries for specified subjects from the JSON file and save to a new file.
	
	Args:
		json_file (str): Path to the source JSON file
		saved_json_file (str): Path where the extracted data will be saved
		clean_data_subjects (list): List of subject identifiers to be extracted
	"""
	data_dict = json.load(open(json_file, 'r'))
	new_data_dict = {}
	counter = 0
	for k, v in data_dict.items():
		if v['subject'] in clean_data_subjects:
			new_data_dict[str(counter)] = v
			counter += 1
	print(f'{counter} subjects are extracted')
	with open(saved_json_file, 'w', encoding='utf-8') as f:
		json.dump(new_data_dict, f, indent=2)


if __name__ == '__main__':
	######################
	# #copy the original json file and add a _0426 suffix first!
	######################
	# Generate json file with the data
	saved_json_file = os.path.join(saved_json_folder, f'{feat_name}.json')
	create_json_with_data_folder(folder_path, saved_json_file)
	_add_features_to_json(saved_json_file,feature_folder_path)
	update_moca_mmse_score(saved_json_file, patient_info_path, saved_json_file)
	_add_disease_info_to_json(saved_json_file, patient_info_path, saved_json_file)
	
    # combine_json_files(saved_json_file, saved_json_file2, saved_json_file3)
	# delete_given_subject_from_json(saved_json_file, ['010_patient'], saved_json_file)
	# check_key_order(saved_json_file)

