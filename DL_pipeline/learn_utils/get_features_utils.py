from DL_pipeline.learn_utils import *
from DL_pipeline.dataset import *
from utils import *
from DL_pipeline.learn_utils.feature_extractor import FeatureExtractor
from DL_pipeline.learn_utils.init import get_model

def summarize_all_subjects_data(json_data):
    """
    Summarize data for all subjects from the JSON file, including cognitive scores.
    
    Args:
        json_data (dict): The loaded JSON data
        
    Returns:
        dict: A dictionary with the following structure:
        {
            'subject1': {
                'datetime1': {
                    'cognitive_scores': {
                        'MoCA': score,
                        'MMSE': score
                    },
                    'tasks': {
                        'task1': {
                            'pics': {
                                'pic1': [indices],
                                'pic2': [indices],
                            }
                        },
                        'task2': {
                            'pics': {...}
                        }
                    }
                },
                'datetime2': {...}
            },
            'subject2': {...}
        }
    """
    result = {}
    
    # Iterate through all entries in the JSON data
    for idx, entry in json_data.items():
        subject = entry['subject']
        datetime = entry['date_time']
        task = entry['task_no']
        pic = entry['pic_no']
        
        # Initialize subject if not exists
        if subject not in result:
            result[subject] = {}
            
        # Initialize datetime if not exists
        if datetime not in result[subject]:
            result[subject][datetime] = {
                'cognitive_scores': {
                    'MoCA': entry['MoCA'],
                    'MMSE': entry['MMSE'],
                },
                'tasks': {},
                'data_type': entry['data_type']
            }
            

        # Initialize task if not exists
        if task not in result[subject][datetime]['tasks']:
            result[subject][datetime]['tasks'][task] = {
                'pics': {}
            }
            
        # Initialize pic if not exists
        if pic not in result[subject][datetime]['tasks'][task]['pics']:
            result[subject][datetime]['tasks'][task]['pics'][pic] = []
            
        # Add index for this pic
        result[subject][datetime]['tasks'][task]['pics'][pic].append(int(idx))
    
    return result


def get_subject_task_indices(subject_dict, subject, task_no, date_time=None):
    all_indices = []

    data_info = subject_dict[subject]
    # If date_time is None, use first datetime in subject dict
    if date_time is None:
        date_time = next(iter(data_info.keys()))
        # Initialize list to store all indices
        task_info = data_info[date_time]['tasks'][task_no]['pics']
        for pic in task_info.keys():
            all_indices.extend(task_info[pic])
        
    # If date_time is "all", collect indices from all datetimes
    elif date_time == "all":
        for dt in data_info.keys():
            task_info = data_info[dt]['tasks'][task_no]['pics']
            for pic in task_info.keys():
                all_indices.extend(task_info[pic])
        # Otherwise get indices just for specified datetime
    else:
        task_info = data_info[date_time]['tasks'][task_no]['pics']
        for pic in task_info.keys():
            all_indices.extend(task_info[pic])
                        
    return all_indices
            
def get_subject_tasks(summary_dict, subject):
    """
    Get all unique tasks for a given subject across all datetimes.
    
    Args:
        summary_dict (dict): The summary dictionary containing all subjects' data
        subject (str): The subject ID to query
        
    Returns:
        dict: A dictionary containing:
            - all_tasks: list of all unique tasks
            - task_details: dictionary with detailed information for each task
    """
    if subject not in summary_dict:
        return {"error": f"Subject {subject} not found"}
    
    all_tasks = set()
    task_details = {}
    
    # Iterate through all datetimes for the subject
    for datetime, data in summary_dict[subject].items():
        for task in data['tasks'].keys():
            all_tasks.add(task)
            
            # Initialize task details if not exists
            if task not in task_details:
                task_details[task] = {
                    'datetimes': [],
                    'pics': {},
                    'total_samples': 0
                }
            
            # Add datetime
            task_details[task]['datetimes'].append(datetime)
            
            # Add pics and their indices
            for pic, indices in data['tasks'][task]['pics'].items():
                if pic not in task_details[task]['pics']:
                    task_details[task]['pics'][pic] = []
                task_details[task]['pics'][pic].extend(indices)
                task_details[task]['total_samples'] += len(indices)
    
    return {
        'all_tasks': sorted(list(all_tasks)),
        'task_details': task_details
    }





def get_features_subject_activity(cfg, model_dict, subject, date_time, data_type, task_no, pic_no=None, save_path=None):
    indices = split_data_json_by_subject_activity(cfg['sliced_trials_json'], subject, data_type, task_no, pic_no)
    _, _, dataloader = select_dataloader(None, None, indices, cfg)
    model = get_model(cfg, model_dict, subject)
    if pic_no is None:
        feats_name = f'{subject}-{date_time}-{data_type}-{task_no}'
    else:
        feats_name = f'{subject}-{date_time}-{data_type}-{task_no}-{pic_no}'
    feature_extractor = FeatureExtractor(model=model, device=cfg['device'], 
                                         feats_name=feats_name,
                                         save_path=save_path)
    features = feature_extractor.extract(dataloader)
    feature_extractor.save()
    return features

def get_all_features(cfg, model_dict, save_path,if_pic_level=False):
    features = {}
    subject_dict = get_subject_activity_dict(cfg['sliced_trials_json'])
    for subject in subject_dict.keys():
        features[subject] = {}
        for activity in subject_dict[subject]:
            date_time, data_type, task_no, pic_no = activity.split('-')
            if if_pic_level is False:
                feats_key = f'{date_time}-{data_type}-{task_no}'
                if feats_key not in features[subject].keys():
                    features[subject][feats_key] = []
                features[subject][feats_key].append(get_features_subject_activity(cfg, model_dict, subject, date_time, data_type, task_no))
            else:
                feats_key = f'{date_time}-{data_type}-{task_no}-{pic_no}'
                if feats_key not in features[subject].keys():
                    features[subject][feats_key] = []   
                features[subject][feats_key].append(get_features_subject_activity(cfg, model_dict, subject, date_time, data_type, task_no, pic_no))
    return features

def load_features(folder):
    features = {}
    for file in os.listdir(folder):
        if file.endswith(".npz"):
            feats_name = file.split(".")[0]
            ### get feats_key
            if len(feats_name.split('-')) == 5:
                subject, date_time, data_type, task_no, pic_no = feats_name.split('-')
                feats_key = f'{date_time}-{data_type}-{task_no}-{pic_no}'
            elif len(feats_name.split('-')) == 4:
                subject, date_time, data_type, task_no = feats_name.split('-')
                feats_key = f'{date_time}-{data_type}-{task_no}'
            else:
                raise ValueError(f"Invalid features name: {feats_name}")
            
            ### get features
            if subject not in features.keys():
                features[subject] = {}
            if feats_key not in features[subject].keys():
                features[subject][feats_key] = []
            features[subject][feats_key].append(np.load(os.path.join(folder, file), allow_pickle=True))
    return features

def group_subject_test_sessions(subjects_dict):
    """
    Group datetimes into test sessions for each subject.
    A test session typically includes a resting task and video tasks, but they have different timestamps.
    
    Args:
        subjects_dict (dict): Dictionary from summarize_all_subjects_data
        
    Returns:
        dict: A dictionary with structure:
        {
            'subject1': {
                'session1': {
                    'resting': {datetime: '2024_12_24_16_03_56', tasks: {...}},
                    'video': {datetime: '2024_12_24_16_10_21', tasks: {...}},
                    'cognitive_scores': {'MoCA': score, 'MMSE': score}
                },
                'session2': {...}
            },
            'subject2': {...}
        }
    """
    result = {}
    
    for subject, datetime_data in subjects_dict.items():
        result[subject] = {}
        
        # Sort datetimes chronologically
        sorted_datetimes = sorted(datetime_data.keys())
        
        # Group by test session (assuming sessions are ordered chronologically)
        current_session = 1
        current_session_data = {'resting': None, 'video': None}
        
        for dt in sorted_datetimes:
            data_type = datetime_data[dt]['data_type']
            
            # If we already have this data type in the current session, start a new session
            if current_session_data[data_type] is not None:
                # Save the completed session
                if current_session_data['resting'] is not None or current_session_data['video'] is not None:
                    result[subject][f'session{current_session}'] = {
                        'resting': current_session_data['resting'],
                        'video': current_session_data['video'],
                        'cognitive_scores': datetime_data[dt]['cognitive_scores']
                    }
                # Start a new session
                current_session += 1
                current_session_data = {'resting': None, 'video': None}
            
            # Add the current datetime data to the session
            current_session_data[data_type] = {
                'datetime': dt,
                'tasks': datetime_data[dt]['tasks']
            }
        
        # Add the last session if it has data
        if current_session_data['resting'] is not None or current_session_data['video'] is not None:
            result[subject][f'session{current_session}'] = {
                'resting': current_session_data['resting'],
                'video': current_session_data['video'],
                'cognitive_scores': datetime_data[sorted_datetimes[-1]]['cognitive_scores']
            }
    
    return result

def batch_extract_features(indices_dict, cfg, model, device, layers, feature_save_path, hook_fn_name='input'):
    """
    Extract features for multiple subjects and tasks in a single batch process.
    
    Args:
        indices_dict: Dictionary mapping (subject, session_key, task) to indices
        cfg: Configuration dictionary
        model: The model to extract features from
        device: Device to run model on
        layers: Dictionary of layers to extract features from
        feature_save_path: Path to save features
        hook_fn_name: Hook function name ('input' or 'output')
        
    Returns:
        Dictionary mapping (subject, session_key, task) to features
    """
    # Collect all indices in a flat list with their metadata
    all_indices = []
    metadata = []
    
    for (subject, session_key, task), indices in indices_dict.items():
        all_indices.extend(indices)
        metadata.extend([(subject, session_key, task)] * len(indices))
    
    if not all_indices:
        return {}
    
    # Create a single dataloader for all indices
    _, _, dataloader = select_dataloader(None, None, all_indices, cfg)
    
    # Extract features in a single batch process
    if layers is None:
        layers = {'final_layer': model.final_layer}
    
    feature_extractor = FeatureExtractor(model, device, "batch_features", layers, feature_save_path)
    all_features = feature_extractor.extract(dataloader, hook_fn_name=hook_fn_name)
    
    # Organize features back by subject, session, and task
    result = {}
    start_idx = 0
    
    # Group indices by metadata to determine feature slices
    grouped_indices = {}
    for i, meta in enumerate(metadata):
        if meta not in grouped_indices:
            grouped_indices[meta] = []
        grouped_indices[meta].append(i)
    
    # Extract features for each group
    for meta, indices in grouped_indices.items():
        subject, session_key, task = meta
        if (subject, session_key) not in result:
            result[(subject, session_key)] = {}
        
        # Create feature dict for this task
        features_dict = {}
        for layer_name, features in all_features.items():
            # Get indices for this metadata and extract corresponding features
            features_dict[layer_name] = features[indices]
        
        result[(subject, session_key)][task] = features_dict
    
    return result 


def get_cfg_indices_with_each_user(cfg):
    data_json = json.load(open(cfg['sliced_trials_json']))
    
    # Initialize subjects_dict with the raw structure
    subjects_dict = summarize_all_subjects_data(data_json)
    
    # Group by test sessions
    subject_sessions = group_subject_test_sessions(subjects_dict)
    
    # Get user_list_ids
    data_set = select_data_set(cfg['data_type'])
    total_users_lists = data_set.total_users_lists
    # test_user_list_idx = cfg['test_user_list_idx']
    # train_user_list_idxs = [i for i in range(len(total_users_lists)) if i != test_user_list_idx]
    
    subject_task_indices = {}
    
    for user_list_idx in range(len(total_users_lists)):
        for subject in total_users_lists[user_list_idx]:
            logger.info(f'Processing subject: [{subject}]')
            if subject not in subject_task_indices:
                subject_task_indices[subject] = {}
            
            # Use the first session that has both resting and video data
            first_complete_session = None
            for session, session_data in subject_sessions[subject].items():
                if session_data['resting'] is not None and session_data['video'] is not None:
                    first_complete_session = session
                    break
            
            if first_complete_session is None:
                logger.warning(f"No complete session found for subject {subject}")
                continue
                
            session_data = subject_sessions[subject][first_complete_session]
            
            # Create a single entry for this session
            session_key = f"session_{first_complete_session}"
            subject_task_indices[subject][session_key] = {
                'cognitive_scores': session_data['cognitive_scores']
            }
            
            # Process both resting and video tasks
            for data_type in ['resting', 'video']:
                if session_data[data_type] is None:
                    continue
                    
                datetime = session_data[data_type]['datetime']
                for task, task_data in session_data[data_type]['tasks'].items():
                    # Skip resting task if specified
                    if task == 'task0' and data_type == 'resting':
                        continue
                        
                    # Process task indices
                    task_indices = []
                    # for pic, indices in task_data['pics'].items():
                    #     task_indices.extend(indices)
                    n_pics = len(task_data['pics'])
                    for i in range(n_pics):
                        task_indices.extend(task_data['pics'][f'pic{i}'])

                    if task not in subject_task_indices[subject][session_key]:
                        subject_task_indices[subject][session_key][task] = {}
                    
                    subject_task_indices[subject][session_key][task]['indices'] = task_indices
                    
    return subject_task_indices

def get_raw_subject_task_indices(subject_task_indices, user_list, intra_task_agg_method, across_task_agg_method, is_features=True):
    """
    Split and reorganize subject task indices into numpy arrays for training and testing.
    
    Returns:
        features: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples, 2) [MoCA, MMSE]
    """
    features_list = []
    labels_list = []
    # Process training data
    for subject in user_list:
        if subject not in subject_task_indices:
            logger.warning(f'Subject {subject} not found in subject_task_indices')
            continue

        # Get the first session key (should be 'session_1' or similar)
        if len(subject_task_indices[subject].keys()) == 0:
            logger.warning(f'Subject {subject} has no sessions')
            continue
            
        first_session = next(iter(subject_task_indices[subject].keys()))

        cognitive_scores = subject_task_indices[subject][first_session]['cognitive_scores']
        
        subject_task_features = []
        # for task in subject_task_indices[subject][first_session].keys():
        for i in range(10):		## 保证task的顺序！
            task = f'task{i}'	
            if task != 'cognitive_scores':
                if is_features:
                    # Get features for this task
                    task_features = subject_task_indices[subject][first_session][task]['features']
                    if isinstance(task_features, dict):
                        # If features is a dictionary, concatenate all features
                        task_features = np.concatenate([feat for feat in task_features.values()], axis=1)
                        # n_slice, n_dim
                        task_features = aggregate_features_within_task(task_features, method=intra_task_agg_method)					
                        # n_pics, n_dim
                    subject_task_features.append(task_features)
                else:
                    task_indices = subject_task_indices[subject][first_session][task]['indices']
                    task_indices = aggregate_features_within_task(task_indices, method=intra_task_agg_method)
                    subject_task_features.append(task_indices)
                # Repeat cognitive scores for each sample in task_features
        subject_task_features = aggregate_features_across_tasks(subject_task_features, method=across_task_agg_method)
        features_list.append(subject_task_features)
        labels_list.append([cognitive_scores['MoCA'], cognitive_scores['MMSE'], cognitive_scores['MoCA_taskscore'], cognitive_scores['MMSE_taskscore']])

    return features_list, labels_list