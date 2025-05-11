# Installation Guide
Clone repository:

```
git clone https://github.com/EasyCog/EasyCog-Benchmark.git
cd EasyCog-Benchmark
```

Create [Anaconda](https://www.anaconda.com/products/distribution) environment:
```
conda create -n EasyCog python==3.9.19
conda activate EasyCog
```

Install other requirements:
```
pip install -r requirements.txt
```

Install EEGLab following ```https://github.com/sccn/eeglab```

Install the required EEGLab Plugins, and put them in the `EEGLab_path/plugins/`
```
https://github.com/sccn/clean_rawdata
https://github.com/sccn/dipfit
https://github.com/sccn/ICLabel
https://github.com/sccn/EEG-BIDS
https://github.com/sccn/firfilt
```

Make sure you modify the path to EEGlab in `data_processing/process_eeglab.py` as follows:
```
path2eeglab = 'path/to/eeglab'
path2data_processing = 'path/to/data_processing/folder'
```