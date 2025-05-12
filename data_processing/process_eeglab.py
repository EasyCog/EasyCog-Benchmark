"""
EEGLAB processing module for EEG data analysis.
This module provides functions to interface with EEGLAB through Octave.
"""

from oct2py import octave
import os
import sys
import numpy as np
from data_processing.display import *

# Define paths
path2eeglab = 'path/to/eeglab'
path2data_processing = 'path/to/data_processing/folder'



def init_eeg_lab():
    """
    Initialize EEGLAB by adding necessary paths and loading required packages.
    This function sets up the environment for EEGLAB processing.
    """
    # Add EEGLAB function paths
    octave.addpath(path2eeglab + '/functions/guifunc')
    octave.addpath(path2eeglab + '/functions/popfunc')
    octave.addpath(path2eeglab + '/functions/adminfunc')
    octave.addpath(path2eeglab + '/functions/sigprocfunc')
    octave.addpath(path2eeglab + '/functions/miscfunc')
    
    # Add plugin paths
    octave.addpath(path2eeglab + '/plugins/dipfit')
    octave.addpath(path2eeglab + '/plugins/clean_rawdata')
    octave.addpath(path2data_processing + '/')
    
    # Load required Octave packages
    octave.eval('pkg load signal')
    octave.eval('pkg load statistics')

def eeg_optimization(data, fs):
    """
    Perform EEG data optimization.
    
    Args:
        data (numpy.ndarray): Input EEG data
        fs (float): Sampling frequency
        
    Returns:
        numpy.ndarray: Optimized EEG data
    """
    init_eeg_lab()
    return octave.eeg_optimisation(data, fs)

def cal_DTF(data, low_freq, high_freq, p, fs):
    """
    Calculate Directed Transfer Function (DTF) for EEG data.
    
    Args:
        data (numpy.ndarray): Input EEG data
        low_freq (float): Lower frequency bound
        high_freq (float): Upper frequency bound
        p (int): Model order
        fs (float): Sampling frequency
        
    Returns:
        numpy.ndarray: DTF results
    """
    octave.addpath(path2data_processing + '/')
    return octave.DTF(data, low_freq, high_freq, p, fs)

    

