import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DL_pipeline.learn_utils import *
from DL_pipeline.regression_utils import *
from DL_pipeline.Cog_Regression import *
import json
# from utils.subject_split import *
from scipy.stats import linregress
from sklearn import manifold
from utils.subject_split import *

param_group = [
        {
        'json_file': f'/home/mmWave_group/EasyCog/data_json_files/proc_compare_json/asreog_filter_order3_clean_data.json',
        'log_file': f'logs/raw_feat_regression/20250416_regression_metrics_clean_data_videorest_mean_PCA',
        'if_rest_only': False,
        'if_mean': True,        # Fixed
        'best_mean_video': False,   # True is useless
        'if_PCA': True,
        'data_type': 'clean',
    },        
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='clean_0426', help='data type')
    parser.add_argument('--log_file', type=str, default='logs/raw_feat_regression/20250416_regression_metrics_clean_data_videorest_mean_PCA', help='log file')
    parser.add_argument('--if_PCA', type=str_to_bool, default=True, help='if PCA')
    parser.add_argument('--json_file', type=str, default='data_json_files/proc_compare_json/asreog_filter_order3_clean_data.json', help='json file')
    parser.add_argument('--if_rest_only', type=str_to_bool, default=False, help='if rest only')
    parser.add_argument('--if_mean', type=str_to_bool, default=True, help='if mean')
    return parser.parse_args()

def normalize(data):
    # return (data - np.mean(data)) / np.std(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_regression_io_with_each_user(json_file):
    data_json = json.load(open(json_file))
    
    # Initialize subjects_dict with the raw structure
    subjects_dict = summarize_all_subjects_data(data_json)
    
    # Group by test sessions
    subject_sessions = group_subject_test_sessions(subjects_dict)
    
    # Get user_list_ids
    # test_user_list_idx = 4
    # train_user_list_idxs = [i for i in range(len(total_users_lists)) if i != test_user_list_idx]
    
    subject_task_indices = {}

    # First, collect all indices organized by subject, session, and task
    indices_dict = {}
    
    for user_list_idx in range(len(total_users_lists)):
        for subject in total_users_lists[user_list_idx]:
            if subject not in subject_task_indices:
                subject_task_indices[subject] = {}
            
            # Use the first session that has both resting and video data
            first_complete_session = None
            if subject not in subject_sessions.keys():
                continue

            for session, session_data in subject_sessions[subject].items():
                if session_data['resting'] is not None and session_data['video'] is not None:
                    first_complete_session = session
                    break

            if first_complete_session is None:
                # delete the key
                del subject_sessions[subject]
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
                    for pic, indices in task_data['pics'].items():
                        task_indices.extend(indices)
                        
                    if task not in subject_task_indices[subject][session_key]:
                        subject_task_indices[subject][session_key][task] = {}
                    
                    subject_task_indices[subject][session_key][task]['indices'] = task_indices
                    
                    # Add to indices_dict for batch processing
                    indices_dict[(subject, session_key, task)] = task_indices
    
    return subject_task_indices


class OurAdaboost:
    def __init__(self, C=1, lr=0.1, kernel='linear', deg=3, gamma='auto',coef0=0):
        self.alpha = []
        self.models = []
        self.err = []
        self.learning_rate = lr
        self.C = C
        self.kernel = kernel
        self.deg = deg
        self.gamma = gamma
        self.coef0 = coef0
    
    def fit(self, X, y):
        n_samples = X[0].shape[0]
        w = np.ones(n_samples) / n_samples
        epsilon = np.finfo(w.dtype).eps
        for i in range(len(X)):
            print("Fitting model %d" % i)
            w = np.clip(w, a_min=epsilon, a_max=None)
            model = SVR(kernel=self.kernel,C=self.C, gamma=self.gamma, degree=self.deg, coef0=self.coef0)
            model.fit(X[i].reshape(n_samples,-1), y, sample_weight=w)
            y_pred = model.predict(X[i].reshape(n_samples,-1))
            # Calculate the regression error rather than classification error
            print("STD of y_pred: ", np.std(y_pred))
            err = np.abs(y_pred - y)
            print("Mean Error: ", np.mean(err))
            sample_mask = w > 0
            masked_sample_weight = w[sample_mask]
            masked_err = err[sample_mask]

            err_max = masked_err.max()
            if err_max != 0:
                masked_err /= err_max
            masked_err  **= 2

            estimator_err = (masked_sample_weight * masked_err).sum()
            self.err.append(estimator_err)

            if estimator_err <= 0:
                self.alpha.append(1)
                self.models.append(model)
                break
                
            beta = estimator_err / (1 - estimator_err)
            estimator_weight = self.learning_rate * np.log(1. / beta)
            print("Estimator Weight: ", estimator_weight)
            w[sample_mask] *= np.power(beta , (1. - masked_err)*self.learning_rate)
            
            self.alpha.append(estimator_weight)
            self.models.append(model)
            w /= w.sum()
    
    def predict(self, X, mode='median'):
        if mode == 'weighted':
            return self._get_weighted_predict(X)
        else:
            return self._get_median_predict(X)
    
    def _get_weighted_predict(self,X):
        n_samples = X[0].shape[0]
        # give the predicted regresson result
        y_pred = np.zeros(n_samples)
        for i in range(len(X)):
            y_pred += self.alpha[i]/np.sum(self.alpha) * self.models[i].predict(X[i].reshape(n_samples,-1))
        return y_pred

    def _get_median_predict(self,X):
        n_samples = X[0].shape[0]
        y_pred = np.array([est.predict(X[i].reshape(n_samples,-1)) for i, est in enumerate(self.models)]).T

        # sort the predictions
        sorted_idx = np.argsort(y_pred, axis=1)

        # find index of median prediction for each sample
        weight_cdf = np.cumsum(np.array(self.alpha)[sorted_idx], axis=1, dtype=np.float64)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(n_samples), median_idx]

        return y_pred[np.arange(n_samples), median_estimators]
        


if __name__ == '__main__':
    args = parse_args()

    log_file = args.log_file
    if_rest_only = args.if_rest_only
    if_mean = args.if_mean
    # best_mean_video = param['best_mean_video']
    best_mean_video = False
    if_PCA = args.if_PCA

    # json_file = '/home/mmWave_group/EasyCog/data_json_files/proc_compare_json/asreog_filter_order3_clean_data_0426.json'
    # log_file = 'logs/Cog_Baselines/Handfeat_Adaboost/clean_0426/log_adaboost_five_folds.out'
    # os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # data_type = 'clean_0426'
    # if_rest_only = False
    # if_mean = True
    # if_PCA = True
    
    if args.data_type == 'clean':
        total_users_lists = Clean_Subjects.total_users_lists
        json_file='/home/mmWave_group/EasyCog/data_json_files/proc_compare_json/asreog_filter_order3_clean_data.json'
    elif args.data_type == 'all_noisy':
        total_users_lists = All_Subjects.total_users_lists
        json_file='/home/mmWave_group/EasyCog/data_json_files/proc_compare_json/asreog_filter_order3_all_data.json'
    elif args.data_type == 'clean_0426':
        total_users_lists = Score_Balanced_Subjects_Clean.total_users_lists
        json_file='/home/mmWave_group/EasyCog/data_json_files/proc_compare_json/asreog_filter_order3_clean_data_0426.json'
    elif args.data_type == 'all_0426':
        total_users_lists = Score_Balanced_Subjects_All.total_users_lists
        json_file='/home/mmWave_group/EasyCog/data_json_files/proc_compare_json/asreog_filter_order3_all_data_0426.json'
    
    logger.info(f'data_type: {args.data_type}, json_file: {json_file}, if_rest_only: {if_rest_only}, if_mean: {if_mean}, if_PCA: {if_PCA}')

    adaboost_moca_label = []
    adaboost_mmse_label = []
    adaboost_moca_output = []
    adaboost_mmse_output = []

    adaboost_moca_valid_label = []
    adaboost_mmse_valid_label = []
    adaboost_moca_valid_output = []
    adaboost_mmse_valid_output = []

    our_adaboost_mean_moca_label = []
    our_adaboost_mean_mmse_label = []
    our_adaboost_mean_moca_output = []
    our_adaboost_mean_mmse_output = []

    our_adaboost_weighted_moca_label = []
    our_adaboost_weighted_mmse_label = []
    our_adaboost_weighted_moca_output = []
    our_adaboost_weighted_mmse_output = []

    for i in range(len(total_users_lists)):
        # Load the json file
        data_dict = json.load(open(json_file, 'r'))
        subject_task_indices = get_regression_io_with_each_user(json_file)

        test_user_list_idx = i
        valid_user_list_idx = (i + 1) % 10
        train_user_list_idxs = [i for i in range(len(total_users_lists)) if (i != test_user_list_idx and i != valid_user_list_idx)]
        test_user_list = total_users_lists[test_user_list_idx]
        valid_user_list = total_users_lists[valid_user_list_idx]
        


        # train user list is all combined
        train_user_list = []
        for user_list in total_users_lists:
            if user_list != test_user_list and user_list != valid_user_list:
                train_user_list.extend(user_list)

        logger.info(f'test_user_list: {test_user_list}, valid_user_list: {valid_user_list}')
        logger.info(f'train_user_list_idxs: {train_user_list}')

        # Prepare the data and labels


        features_fractal_train = []
        features_complexity_train = []
        features_micro_state_train = []
        features_frequency_train = []
        features_spatial_train = []
        features_entropy_train = []
        features_time_domain_train = []
        label_MMSE_train = []
        label_MoCA_train = []


        features_fractal_test = []
        features_complexity_test = []
        features_micro_state_test = []
        features_frequency_test = []
        features_spatial_test = []
        features_entropy_test = []
        features_time_domain_test = []
        label_MMSE_test = []
        label_MoCA_test = []

        features_fractal_valid = []
        features_complexity_valid = []
        features_micro_state_valid = []
        features_frequency_valid = []
        features_spatial_valid = []
        features_entropy_valid = []
        features_time_domain_valid = []
        label_MMSE_valid = []
        label_MoCA_valid = []

        cnt = 0
        for k, v in subject_task_indices.items():
            cnt += 1
            print(f'Processing {cnt}/{len(subject_task_indices.keys())} subjects')
            if v == {}:
                continue

            moca = v['session_session1']['cognitive_scores']['MoCA']
            mmse = v['session_session1']['cognitive_scores']['MMSE']

            
            rest_indices = v['session_session1']['task9']['indices']


            features_fractal = []
            features_complexity = []
            features_micro_state = []
            features_frequency = []
            features_spatial = []
            features_entropy = []
            features_time_domain = []

            label_MoCA = []
            label_MMSE = []
            labels = []
            for idx in rest_indices:
                sliced_data = data_dict[str(idx)]
                feat_path = sliced_data['feat_file']
                feat_list = sliced_data['features']
            
                label_MMSE.append(mmse)
                label_MoCA.append(moca)

                feat_data = np.load(feat_path)

                fractal_feat = np.array([])
                complexity_feat = np.array([])
                micro_state_feat = np.array([])
                frequency_feat = np.array([])
                spatial_feat = np.array([])
                entropy_feat = np.array([])
                time_domain_feat = np.array([])

                for f in feat_list:
                    # fractal_feat
                    if f in ['peak_value', 'rectification_avg', 'effecteive_value', 'kurtosis', 'skewness', 'margin', 'form_factor', 'impulse_factor','crest_factor']:
                        fractal_feat = np.hstack((fractal_feat, feat_data[f].flatten())) if fractal_feat.size else feat_data[f].flatten()
                    # complexity_feat
                    elif f in ['signal_complexity']:
                        complexity_feat = np.hstack((complexity_feat, feat_data[f].flatten())) if complexity_feat.size else feat_data[f].flatten()
                    # # micro_state_feat
                    elif f in ['micro_states_occurrences', 'micro_states_transition', 'micro_states_dist', 'micro_states_center']:
                        micro_state_feat = np.hstack((micro_state_feat, feat_data[f].flatten())) if micro_state_feat.size else feat_data[f].flatten()
                    # frequency_feat
                    elif f in ['delta_ratio', 'theta_ratio', 'alpha_ratio', 'beta_ratio', 'gamma_ratio',
                'beta_to_alpha_ratio', 'theta_to_alpha_ratio', 'theta_alpha_to_beta_ratio',
                'theta_to_beta_ratio', 'theta_alpha_to_alpha_beta_ratio', 'gamma_to_delta_ratio',
                'gamma_beta_to_delta_alpha_ratio']:
                        frequency_feat = np.hstack((frequency_feat, feat_data[f].flatten())) if frequency_feat.size else feat_data[f].flatten()
                    # spatial_feat
                    elif f in ['channel_correlation', 'mutual_information']:
                        spatial_feat = np.hstack((spatial_feat, feat_data[f].flatten())) if spatial_feat.size else feat_data[f].flatten()
                    # entropy_feat
                    elif f in ['entropy']:
                        entropy_feat = np.hstack((entropy_feat, feat_data[f].flatten())) if entropy_feat.size else feat_data[f].flatten()
                    # time_domain_feat
                    elif f in ['mean', 'std', 'median', 'peak_value']:
                        time_domain_feat = np.hstack((time_domain_feat, feat_data[f].flatten())) if time_domain_feat.size else feat_data[f].flatten()
            
                # list of features, len(list) = n_sliced_files, each element is a 1d array with length = dim_features
                features_fractal.append(fractal_feat)
                features_complexity.append(complexity_feat)
                features_micro_state.append(micro_state_feat)
                features_frequency.append(frequency_feat)
                features_spatial.append(spatial_feat)
                features_entropy.append(entropy_feat)
                features_time_domain.append(time_domain_feat)

            # normalize the features to [0,1]
            features_fractal = normalize(np.array(features_fractal))
            features_complexity = normalize(np.array(features_complexity))
            features_micro_state = normalize(np.array(features_micro_state))
            features_frequency = normalize(np.array(features_frequency))
            features_spatial = normalize(np.array(features_spatial))
            features_entropy = normalize(np.array(features_entropy))
            features_time_domain = normalize(np.array(features_time_domain))
        
            if if_mean:
                # ndarray with shape (1, dim_features)
                features_fractal = np.mean(np.array(features_fractal), axis=0, keepdims=True)
                features_complexity = np.mean(np.array(features_complexity), axis=0, keepdims=True)
                features_micro_state = np.mean(np.array(features_micro_state), axis=0, keepdims=True)
                features_frequency = np.mean(np.array(features_frequency), axis=0, keepdims=True)
                features_spatial = np.mean(np.array(features_spatial), axis=0, keepdims=True)
                features_entropy = np.mean(np.array(features_entropy), axis=0, keepdims=True)
                features_time_domain = np.mean(np.array(features_time_domain), axis=0, keepdims=True)
            
            if not if_rest_only:
                features_fractal_video = []
                features_complexity_video = []
                features_micro_state_video = []
                features_frequency_video = []
                features_spatial_video = []
                features_entropy_video = []
                features_time_domain_video = []
                for i in range(9):
                    video_indices = (v['session_session1'][f'task{i}']['indices'])
                    features_fractal_task = []
                    features_complexity_task = []
                    features_micro_state_task = []
                    features_frequency_task = []
                    features_spatial_task = []
                    features_entropy_task = []
                    features_time_domain_task = []
                    for idx in video_indices:
                        sliced_data = data_dict[str(idx)]
                        feat_path = sliced_data['feat_file']
                        feat_list = sliced_data['features']
                
                        feat_data = np.load(feat_path)

                        fractal_feat = np.array([])
                        complexity_feat = np.array([])
                        micro_state_feat = np.array([])
                        frequency_feat = np.array([])
                        spatial_feat = np.array([])
                        entropy_feat = np.array([])
                        time_domain_feat = np.array([])

                        for f in feat_list:
                            # fractal_feat
                            if f in ['peak_value', 'rectification_avg', 'effecteive_value', 'kurtosis', 'skewness', 'margin', 'form_factor', 'impulse_factor','crest_factor']:
                                fractal_feat = np.hstack((fractal_feat, feat_data[f].flatten())) if fractal_feat.size else feat_data[f].flatten()
                            # complexity_feat
                            elif f in ['signal_complexity']:
                                complexity_feat = np.hstack((complexity_feat, feat_data[f].flatten())) if complexity_feat.size else feat_data[f].flatten()
                            # # micro_state_feat
                            elif f in ['micro_states_occurrences', 'micro_states_transition', 'micro_states_dist', 'micro_states_center']:
                                micro_state_feat = np.hstack((micro_state_feat, feat_data[f].flatten())) if micro_state_feat.size else feat_data[f].flatten()
                            # frequency_feat
                            elif f in ['delta_ratio', 'theta_ratio', 'alpha_ratio', 'beta_ratio', 'gamma_ratio',
                        'beta_to_alpha_ratio', 'theta_to_alpha_ratio', 'theta_alpha_to_beta_ratio',
                        'theta_to_beta_ratio', 'theta_alpha_to_alpha_beta_ratio', 'gamma_to_delta_ratio',
                        'gamma_beta_to_delta_alpha_ratio']:
                                frequency_feat = np.hstack((frequency_feat, feat_data[f].flatten())) if frequency_feat.size else feat_data[f].flatten()
                            # spatial_feat
                            elif f in ['channel_correlation', 'mutual_information']:
                                spatial_feat = np.hstack((spatial_feat, feat_data[f].flatten())) if spatial_feat.size else feat_data[f].flatten()
                            # entropy_feat
                            elif f in ['entropy']:
                                entropy_feat = np.hstack((entropy_feat, feat_data[f].flatten())) if entropy_feat.size else feat_data[f].flatten()
                            # time_domain_feat
                            elif f in ['mean', 'std', 'median', 'peak_value']:
                                time_domain_feat = np.hstack((time_domain_feat, feat_data[f].flatten())) if time_domain_feat.size else feat_data[f].flatten()
                    

                        features_fractal_task.append(fractal_feat)
                        features_complexity_task.append(complexity_feat)
                        features_micro_state_task.append(micro_state_feat)
                        features_frequency_task.append(frequency_feat)
                        features_spatial_task.append(spatial_feat)
                        features_entropy_task.append(entropy_feat)
                        features_time_domain_task.append(time_domain_feat)

                    if if_mean:
                        # ndarray with shape (dim_features)
                        features_fractal_task = np.mean(np.array(features_fractal_task), axis=0, keepdims=False)
                        features_complexity_task = np.mean(np.array(features_complexity_task), axis=0, keepdims=False)
                        features_micro_state_task = np.mean(np.array(features_micro_state_task), axis=0, keepdims=False)
                        features_frequency_task = np.mean(np.array(features_frequency_task), axis=0, keepdims=False)
                        features_spatial_task = np.mean(np.array(features_spatial_task), axis=0, keepdims=False)
                        features_entropy_task = np.mean(np.array(features_entropy_task), axis=0, keepdims=False)
                        features_time_domain_task = np.mean(np.array(features_time_domain_task), axis=0, keepdims=False)
                    else:
                        raise ValueError("if_mean must be True")
                    
                    features_fractal_video.append(features_fractal_task)
                    features_complexity_video.append(features_complexity_task)
                    features_micro_state_video.append(features_micro_state_task)
                    features_frequency_video.append(features_frequency_task)
                    features_spatial_video.append(features_spatial_task)
                    features_entropy_video.append(features_entropy_task)
                    features_time_domain_video.append(features_time_domain_task)
                
                # ndarray with shape (n_tasks, dim_features)
                features_fractal_video = normalize(np.array(features_fractal_video))
                features_complexity_video = normalize(np.array(features_complexity_video))
                features_micro_state_video = normalize(np.array(features_micro_state_video))
                features_frequency_video = normalize(np.array(features_frequency_video))
                features_spatial_video = normalize(np.array(features_spatial_video))
                features_entropy_video = normalize(np.array(features_entropy_video))
                features_time_domain_video = normalize(np.array(features_time_domain_video))

                if best_mean_video:
                    features_fractal_video = np.mean(features_fractal_video, axis=0, keepdims=True)
                    features_complexity_video = np.mean(features_complexity_video, axis=0, keepdims=True)
                    features_micro_state_video = np.mean(features_micro_state_video, axis=0, keepdims=True)
                    features_frequency_video = np.mean(features_frequency_video, axis=0, keepdims=True)
                    features_spatial_video = np.mean(features_spatial_video, axis=0, keepdims=True)
                    features_entropy_video = np.mean(features_entropy_video, axis=0, keepdims=True)
                    features_time_domain_video = np.mean(features_time_domain_video, axis=0, keepdims=True)

                # ndarray with shape (n_tasks + 1, dim_features)
                features_fractal = np.concatenate([features_fractal, features_fractal_video], axis=0)
                features_complexity = np.concatenate([features_complexity, features_complexity_video], axis=0)
                features_micro_state = np.concatenate([features_micro_state, features_micro_state_video], axis=0)
                features_frequency = np.concatenate([features_frequency, features_frequency_video], axis=0)
                features_spatial = np.concatenate([features_spatial, features_spatial_video], axis=0)
                features_entropy = np.concatenate([features_entropy, features_entropy_video], axis=0)
                features_time_domain = np.concatenate([features_time_domain, features_time_domain_video], axis=0)

            if k in test_user_list:
                features_fractal_test.append(np.array(features_fractal))
                features_complexity_test.append(np.array(features_complexity))
                features_micro_state_test.append(np.array(features_micro_state))
                features_frequency_test.append(np.array(features_frequency))
                features_spatial_test.append(np.array(features_spatial))
                features_entropy_test.append(np.array(features_entropy))
                features_time_domain_test.append(np.array(features_time_domain))
                
                label_MMSE_test.append(np.array(label_MMSE))
                label_MoCA_test.append(np.array(label_MoCA))
            elif k in train_user_list:
                features_fractal_train.append(np.array(features_fractal))
                features_complexity_train.append(np.array(features_complexity))
                features_micro_state_train.append(np.array(features_micro_state))
                features_frequency_train.append(np.array(features_frequency))
                features_spatial_train.append(np.array(features_spatial))
                features_entropy_train.append(np.array(features_entropy))
                features_time_domain_train.append(np.array(features_time_domain))

                label_MMSE_train.append(np.array(label_MMSE))
                label_MoCA_train.append(np.array(label_MoCA))
            elif k in valid_user_list:
                features_fractal_valid.append(np.array(features_fractal))
                features_complexity_valid.append(np.array(features_complexity))
                features_micro_state_valid.append(np.array(features_micro_state))
                features_frequency_valid.append(np.array(features_frequency))
                features_spatial_valid.append(np.array(features_spatial))
                features_entropy_valid.append(np.array(features_entropy))
                features_time_domain_valid.append(np.array(features_time_domain))

                label_MMSE_valid.append(np.array(label_MMSE))
                label_MoCA_valid.append(np.array(label_MoCA))
                

        features_fractal_train = np.array(features_fractal_train)
        features_complexity_train = np.array(features_complexity_train)
        features_micro_state_train = np.array(features_micro_state_train)
        features_frequency_train = np.array(features_frequency_train)
        features_spatial_train = np.array(features_spatial_train)
        features_entropy_train = np.array(features_entropy_train)
        features_time_domain_train = np.array(features_time_domain_train)
        features_fractal_test = np.array(features_fractal_test)
        features_complexity_test = np.array(features_complexity_test)
        features_micro_state_test = np.array(features_micro_state_test)
        features_frequency_test = np.array(features_frequency_test)
        features_spatial_test = np.array(features_spatial_test)
        features_entropy_test = np.array(features_entropy_test)
        features_time_domain_test = np.array(features_time_domain_test)
        features_fractal_valid = np.array(features_fractal_valid)
        features_complexity_valid = np.array(features_complexity_valid)
        features_micro_state_valid = np.array(features_micro_state_valid)
        features_frequency_valid = np.array(features_frequency_valid)
        features_spatial_valid = np.array(features_spatial_valid)
        features_entropy_valid = np.array(features_entropy_valid)
        features_time_domain_valid = np.array(features_time_domain_valid)


        print("Dataset loaded")
        # X_train = [np.concatenate(features_fractal_train, axis=0),
        #             np.concatenate(features_complexity_train, axis=0),
        #             np.concatenate(features_micro_state_train, axis=0),
        #             np.concatenate(features_frequency_train, axis=0),
        #                 np.concatenate(features_spatial_train, axis=0),
        #                 np.concatenate(features_entropy_train, axis=0),
        #                     np.concatenate(features_time_domain_train, axis=0)]
        X_train = [features_fractal_train.reshape(features_fractal_train.shape[0], -1),
                    features_complexity_train.reshape(features_complexity_train.shape[0], -1),
                    features_micro_state_train.reshape(features_micro_state_train.shape[0], -1),
                    features_frequency_train.reshape(features_frequency_train.shape[0], -1),
                    features_spatial_train.reshape(features_spatial_train.shape[0], -1),
                    features_entropy_train.reshape(features_entropy_train.shape[0], -1),
                    features_time_domain_train.reshape(features_time_domain_train.shape[0], -1)]

        X_all_train = [np.array(features_fractal_train),
                    np.array(features_complexity_train),
                    np.array(features_micro_state_train),
                    np.array(features_frequency_train),
                    np.array(features_spatial_train),
                    np.array(features_entropy_train),
                    np.array(features_time_domain_train)]
        X_all_train = np.concatenate(X_all_train, axis=2)
        X_all_train = X_all_train.reshape(X_all_train.shape[0], -1)


        if if_mean:
            y_train = [np.array(label_MoCA_train)[:,0], np.array(label_MMSE_train)[:,0]]
        else:
            y_train = [np.concatenate(label_MoCA_train, axis=0), np.concatenate(label_MMSE_train, axis=0)]

        # X_test = [features_fractal_test,
        #            features_complexity_test,
        #             features_micro_state_test,
        #              features_frequency_test,
        #                features_spatial_test,
        #                  features_entropy_test,
        #                    features_time_domain_test]
        # X_all_test = np.concatenate(X_test, axis=1)
        # y_test = [label_MoCA_test, label_MMSE_test]

        # X_test = [np.concatenate(features_fractal_test, axis=0),
        #         np.concatenate(features_complexity_test, axis=0),
        #         np.concatenate(features_micro_state_test, axis=0),
        #         np.concatenate(features_frequency_test, axis=0),
        #         np.concatenate(features_spatial_test, axis=0),
        #         np.concatenate(features_entropy_test, axis=0),
        #         np.concatenate(features_time_domain_test, axis=0)]
        X_test = [features_fractal_test.reshape(features_fractal_test.shape[0], -1),
                    features_complexity_test.reshape(features_complexity_test.shape[0], -1),
                    features_micro_state_test.reshape(features_micro_state_test.shape[0], -1),
                    features_frequency_test.reshape(features_frequency_test.shape[0], -1),
                    features_spatial_test.reshape(features_spatial_test.shape[0], -1),
                    features_entropy_test.reshape(features_entropy_test.shape[0], -1),
                    features_time_domain_test.reshape(features_time_domain_test.shape[0], -1)]
        X_valid = [features_fractal_valid.reshape(features_fractal_valid.shape[0], -1),
                    features_complexity_valid.reshape(features_complexity_valid.shape[0], -1),
                    features_micro_state_valid.reshape(features_micro_state_valid.shape[0], -1),
                    features_frequency_valid.reshape(features_frequency_valid.shape[0], -1),
                    features_spatial_valid.reshape(features_spatial_valid.shape[0], -1),
                    features_entropy_valid.reshape(features_entropy_valid.shape[0], -1),
                    features_time_domain_valid.reshape(features_time_domain_valid.shape[0], -1)]
        X_all_test = [np.array(features_fractal_test),
                    np.array(features_complexity_test),
                    np.array(features_micro_state_test),
                    np.array(features_frequency_test),
                    np.array(features_spatial_test),
                    np.array(features_entropy_test),
                    np.array(features_time_domain_test)]
        X_all_valid = [np.array(features_fractal_valid),
                    np.array(features_complexity_valid),
                    np.array(features_micro_state_valid),
                    np.array(features_frequency_valid),
                    np.array(features_spatial_valid),
                    np.array(features_entropy_valid),
                    np.array(features_time_domain_valid)]
        X_all_test = np.concatenate(X_all_test, axis=2)
        X_all_test = X_all_test.reshape(X_all_test.shape[0], -1)
        X_all_valid = np.concatenate(X_all_valid, axis=2)
        X_all_valid = X_all_valid.reshape(X_all_valid.shape[0], -1)
        if if_mean:
            y_test = [np.array(label_MoCA_test)[:,0], np.array(label_MMSE_test)[:,0]]
            y_valid = [np.array(label_MoCA_valid)[:,0], np.array(label_MMSE_valid)[:,0]]
        else:
            y_test = [np.concatenate(label_MoCA_test, axis=0), np.concatenate(label_MMSE_test, axis=0)]
            y_valid = [np.concatenate(label_MoCA_valid, axis=0), np.concatenate(label_MMSE_valid, axis=0)]

        if if_PCA:
            tmp_PCA_dataset = np.vstack((X_all_train, X_all_test, X_all_valid))
            PCA_result = PCA(n_components=0.99).fit_transform(tmp_PCA_dataset)
            train_len, valid_len, test_len = X_all_train.shape[0], X_all_valid.shape[0], X_all_test.shape[0]
            X_all_train = PCA_result[:train_len]
            X_all_test = PCA_result[train_len:train_len+test_len]
            X_all_valid = PCA_result[train_len+test_len:]

        # tsne=manifold.TSNE(n_components=2, random_state=99)
        # all_data = np.vstack((X_all_train, X_all_test))
        # all_label = np.hstack((y_train[0], y_test[0]))
        # all_data_tsne = tsne.fit_transform(all_data)
        # x_min, x_max = all_data_tsne[:, 0].min(0), all_data_tsne[:, 0].max(0)
        # data_norm = (all_data_tsne - x_min) / (x_max - x_min)

        # healthy_indices = np.where(all_label>=26)[0]
        # mild_indices = np.where((all_label<26) & (all_label>=19))[0]
        # dementia_indices = np.where(all_label<19)[0]
        # healthy_data = data_norm[healthy_indices]
        # mild_data = data_norm[mild_indices]
        # dementia_data = data_norm[dementia_indices]

        # plt.figure()
        # plt.scatter(healthy_data[:, 0], healthy_data[:, 1], c='red',label='healthy(>=26)')
        # plt.scatter(mild_data[:, 0], mild_data[:, 1], c='blue',label='MCI(19-25)')
        # plt.scatter(dementia_data[:, 0], dementia_data[:, 1], c='yellow',label='dementia(<19)')
        # plt.xticks([])
        # plt.yticks([])
        # plt.legend()
        # plt.title("t-SNE of clean data")
        # if if_PCA:
        #     plt.savefig('tsne_PCA.png')
        # else:
        #     plt.savefig('tsne_noPCA.png')



        Regression_model_dict = {
            'AdaBoost' : [StandardRegressor_2D, {'model_name':'AdaBoost', 'learning_rate': 1.0, 'n_estimators': 200, 'random_state': 99}],
            # 'OurAdaboost' : [OurAdaboost, {'C':10e6, 'lr': 1.0, 'kernel': 'linear', 'deg': 3, 'gamma': 'auto', 'coef0': 0}],
        }


        # adaboost_regressor = OurAdaboost(C=10e6,lr=1,kernel='linear',deg=3,gamma='auto',coef0=0)
        print("STD of label: ", np.std(y_train[1]))
        for model_name, model_params in Regression_model_dict.items():
            # repeat 3 times per model
            for i in range(1):
                if model_name == 'OurAdaboost':
                    regression_model_MoCA = model_params[0](**model_params[1])
                    regression_model_MMSE = model_params[0](**model_params[1])
                    data_train, data_test, data_valid = X_train,  X_test, X_valid
                else:
                    trainer = model_params[0](**model_params[1])
                    regression_model_MoCA = trainer.model1
                    regression_model_MMSE = trainer.model2
                    data_train, data_test, data_valid = X_all_train, X_all_test, X_all_valid

                
                print(f'MoCA Regression Model: {regression_model_MoCA}')
                # MoCA_regressor = train_regressor(X_train, y_train, regression_model_MoCA, label_name='MoCA')
                regression_model_MoCA.fit(data_train, y_train[0])
                regression_model_MMSE.fit(data_train, y_train[1])


                if model_name == 'OurAdaboost':
                    our_adaboost_mean_moca_label.append(y_test[0])
                    our_adaboost_mean_mmse_label.append(y_test[1])
                    our_adaboost_mean_moca_output.append(regression_model_MoCA.predict(data_test))
                    our_adaboost_mean_mmse_output.append(regression_model_MMSE.predict(data_test))
                else:
                    adaboost_moca_label.append(y_test[0])
                    adaboost_mmse_label.append(y_test[1])
                    adaboost_moca_output.append(regression_model_MoCA.predict(data_test))
                    adaboost_mmse_output.append(regression_model_MMSE.predict(data_test))
                    adaboost_moca_valid_label.append(y_valid[0])
                    adaboost_mmse_valid_label.append(y_valid[1])
                    adaboost_moca_valid_output.append(regression_model_MoCA.predict(data_valid))
                    adaboost_mmse_valid_output.append(regression_model_MMSE.predict(data_valid))
                    logger.info(f'{model_name}: testset {test_user_list_idx}\n \
                                MoCA prediction: {adaboost_moca_output[-1]}\n \
                                MoCA label: {adaboost_moca_label[-1]}\n \
                                MMSE prediction: {adaboost_mmse_output[-1]}\n \
                                MMSE label: {adaboost_mmse_label[-1]}')
                    logger.info(f'{model_name}: validset {valid_user_list_idx}\n \
                                MoCA prediction: {adaboost_moca_valid_output[-1]}\n \
                                MoCA label: {adaboost_moca_valid_label[-1]}\n \
                                MMSE prediction: {adaboost_mmse_valid_output[-1]}\n \
                                MMSE label: {adaboost_mmse_valid_label[-1]}')


                if model_name == 'OurAdaboost':
                    regression_model_MoCA = model_params[0](**model_params[1])
                    regression_model_MMSE = model_params[0](**model_params[1])
                    regression_model_MoCA.fit(data_train, y_train[0])
                    regression_model_MMSE.fit(data_train, y_train[1])

                    our_adaboost_weighted_moca_label.append(y_test[0])
                    our_adaboost_weighted_mmse_label.append(y_test[1])
                    our_adaboost_weighted_moca_output.append(regression_model_MoCA.predict(data_test, 'weighted'))
                    our_adaboost_weighted_mmse_output.append(regression_model_MMSE.predict(data_test, 'weighted'))

    adaboost_moca_label = np.concatenate(adaboost_moca_label, axis=0)
    adaboost_mmse_label = np.concatenate(adaboost_mmse_label, axis=0)
    adaboost_moca_valid_label = np.concatenate(adaboost_moca_valid_label, axis=0)
    adaboost_mmse_valid_label = np.concatenate(adaboost_mmse_valid_label, axis=0)
    # our_adaboost_mean_moca_label = np.concatenate(our_adaboost_mean_moca_label, axis=0)
    # our_adaboost_mean_mmse_label = np.concatenate(our_adaboost_mean_mmse_label, axis=0)
    # our_adaboost_weighted_moca_label = np.concatenate(our_adaboost_weighted_moca_label, axis=0)
    # our_adaboost_weighted_mmse_label = np.concatenate(our_adaboost_weighted_mmse_label, axis=0)

    adaboost_moca_output = np.concatenate(adaboost_moca_output, axis=0)
    adaboost_mmse_output = np.concatenate(adaboost_mmse_output, axis=0)
    adaboost_moca_valid_output = np.concatenate(adaboost_moca_valid_output, axis=0)
    adaboost_mmse_valid_output = np.concatenate(adaboost_mmse_valid_output, axis=0)
    # our_adaboost_mean_moca_output = np.concatenate(our_adaboost_mean_moca_output, axis=0)
    # our_adaboost_mean_mmse_output = np.concatenate(our_adaboost_mean_mmse_output, axis=0)
    # our_adaboost_weighted_moca_output = np.concatenate(our_adaboost_weighted_moca_output, axis=0)
    # our_adaboost_weighted_mmse_output = np.concatenate(our_adaboost_weighted_mmse_output, axis=0)

    test_metrics_MoCA = get_evaluate_metrics(adaboost_moca_output, adaboost_moca_label)
    test_metrics_MMSE = get_evaluate_metrics(adaboost_mmse_output, adaboost_mmse_label)
    valid_metrics_MoCA = get_evaluate_metrics(adaboost_moca_valid_output, adaboost_moca_valid_label)
    valid_metrics_MMSE = get_evaluate_metrics(adaboost_mmse_valid_output, adaboost_mmse_valid_label)

    np.savez(f"{log_file}_results.npz", adaboost_moca_output=adaboost_moca_output, adaboost_mmse_output=adaboost_mmse_output, adaboost_moca_label=adaboost_moca_label, adaboost_mmse_label=adaboost_mmse_label)
    np.savez(f"{log_file}_valid_results.npz", adaboost_moca_valid_output=adaboost_moca_valid_output, adaboost_mmse_valid_output=adaboost_mmse_valid_output, adaboost_moca_valid_label=adaboost_moca_valid_label, adaboost_mmse_valid_label=adaboost_mmse_valid_label)
    record_regression_metrics_to_csv('Adaboost', test_metrics_MoCA, test_metrics_MMSE, 'MMSE', f"{log_file}_five_fold_adaboost.csv")
    record_regression_metrics_to_csv('Adaboost', valid_metrics_MoCA, valid_metrics_MMSE, 'MMSE', f"{log_file}_five_fold_adaboost_valid.csv")

    k_moca, b_moca, r_value_moca, _, _ = linregress(adaboost_moca_output, adaboost_moca_label)
    k_mmse, b_mmse, r_value_mmse, _, _ = linregress(adaboost_mmse_output, adaboost_mmse_label)

    k_moca_valid, b_moca_valid, r_value_moca_valid, _, _ = linregress(adaboost_moca_valid_output, adaboost_moca_valid_label)
    k_mmse_valid, b_mmse_valid, r_value_mmse_valid, _, _ = linregress(adaboost_mmse_valid_output, adaboost_mmse_valid_label)

    mae_moca = np.abs(adaboost_moca_output-adaboost_moca_label).mean()
    mae_mmse = np.abs(adaboost_mmse_output-adaboost_mmse_label).mean()
    mae_moca_valid = np.abs(adaboost_moca_valid_output-adaboost_moca_valid_label).mean()
    mae_mmse_valid = np.abs(adaboost_mmse_valid_output-adaboost_mmse_valid_label).mean()

    logger.info(f'moca mae_valid: {mae_moca_valid}, mmse mae_valid: {mae_mmse_valid}, moca r_valid: {r_value_moca_valid}, mmse r2_valid: {r_value_mmse_valid}')
    logger.info(f'moca mae: {mae_moca}, mmse mae: {mae_mmse}, moca r: {r_value_moca}, mmse r2: {r_value_mmse}')
    plt.figure()
    plt.subplot(1,2,1)
    colors = ['ro', 'bo', 'go', 'yo', 'co']
    for i in range(len(colors)):
        plt.plot(adaboost_moca_output[i*9:i*9+9], adaboost_moca_label[i*9:i*9+9], colors[i])
    plt.plot(adaboost_moca_output, k_moca*adaboost_moca_output+b_moca, '-')
    plt.xlabel('Pred')
    plt.ylabel('GT')
    plt.xlim([0,30])
    plt.ylim([0,30])
    plt.title(f'MoCA Regression, r={round(r_value_moca, 4)}')
    
    plt.subplot(1,2,2)
    colors = ['ro', 'bo', 'go', 'yo', 'co']
    for i in range(len(colors)):
        plt.plot(adaboost_mmse_output[i*9:i*9+9], adaboost_mmse_label[i*9:i*9+9], colors[i])
    plt.plot(adaboost_mmse_output, k_mmse*adaboost_mmse_output+b_mmse, '-')
    plt.xlabel('Pred')
    plt.ylabel('GT')
    plt.xlim([0,30])
    plt.ylim([0,30])
    plt.title(f'MMSE Regression, r={round(r_value_mmse, 4)}')
    plt.savefig(f"{log_file}_five_fold_adaboost_scatter.png")  

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(adaboost_moca_valid_output, adaboost_moca_valid_label, 'ro')
    plt.plot(adaboost_moca_valid_output, k_moca_valid*adaboost_moca_valid_output+b_moca_valid, '-')
    plt.xlabel('Pred')
    plt.ylabel('GT')
    plt.xlim([0,30])
    plt.ylim([0,30])
    plt.title(f'MoCA Regression, r={round(r_value_moca_valid, 4)}')
    
    plt.subplot(1,2,2)
    plt.plot(adaboost_mmse_valid_output, adaboost_mmse_valid_label, 'ro')
    plt.plot(adaboost_mmse_valid_output, k_mmse_valid*adaboost_mmse_valid_output+b_mmse_valid, '-')
    plt.xlabel('Pred')
    plt.ylabel('GT')
    plt.xlim([0,30])
    plt.ylim([0,30])
    plt.title(f'MMSE Regression, r={round(r_value_mmse_valid, 4)}')
    plt.savefig(f"{log_file}_five_fold_adaboost_valid_scatter.png")
    
    


        # test_metrics_MoCA = get_evaluate_metrics(our_adaboost_mean_moca_output, our_adaboost_mean_moca_label)
        # test_metrics_MMSE = get_evaluate_metrics(our_adaboost_mean_mmse_output, our_adaboost_mean_mmse_label)
        # record_regression_metrics_to_csv('OurAdaboost', test_metrics_MoCA, test_metrics_MMSE, 'MMSE', f"{log_file}_five_fold_our_adaboost_median.csv")

        # test_metrics_MoCA = get_evaluate_metrics(our_adaboost_weighted_moca_output, our_adaboost_weighted_moca_label)
        # test_metrics_MMSE = get_evaluate_metrics(our_adaboost_weighted_mmse_output, our_adaboost_weighted_mmse_label)
        # record_regression_metrics_to_csv('OurAdaboost', test_metrics_MoCA, test_metrics_MMSE, 'MMSE', f"{log_file}_five_fold_our_adaboost_weighted.csv")




