# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# training CNN using pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os


hd5_1 ="/kaggle/input/chunk1/chunk1.hdf5"
csv_file_1 = "/kaggle/input/chunk1/chunk1.csv"
hdf5_2 =  "/kaggle/input/stead-chunk-2/chunk2.hdf5"
csv_file_2 = "/kaggle/input/stead-chunk-2/chunk2.csv"
chunk_1 = pd.read_csv(csv_file_1)
chunk_2 = pd.read_csv(csv_file_2)
full_csv = pd.concat([chunk_1,chunk_2])

num_cols = ['receiver_latitude', 'receiver_longitude', 'receiver_elevation_m', 'p_arrival_sample',
            'p_weight', 'p_travel_sec', 's_arrival_sample', 's_weight', 'source_origin_uncertainty_sec',
            'source_latitude', 'source_longitude', 'source_error_sec', 'source_gap_deg',
            'source_horizontal_uncertainty_km', 'source_depth_km', 'source_depth_uncertainty_km',
            'source_magnitude', 'source_distance_deg', 'source_distance_km', 'back_azimuth_deg',
            'snr_db', 'coda_end_sample']

# %%
full_csv[num_cols].hist(bins=50, figsize=(20,15))
plt.show()
plt.savefig('histogram' +'.png')

# %%
import seaborn as sns
# Check the correlation between numerical columns
corr_matrix = full_csv[num_cols].corr()
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.show()
plt.savefig('co_rel' + '.png')

# %%
# Check the distribution of categorical columns
cat_cols = ['network_code', 'receiver_type', 'p_status', 's_status',
            'source_magnitude_type', 'source_magnitude_author', 'trace_category']
for col in cat_cols:
    plt.figure(figsize=(12,6))
    df[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.show()
    plt.savefig(col + '.png')

# %%
# Print the cleaned dataset
print('Cleaned dataset:')
df.head()
print(f"Shape: {df.shape}")



# %%
def waveform_spectrogram_plot(signal_path,signal_index,signal_list):
    dtfl = h5py.File(signal_path, 'r') # find the signal file
    dataset = dtfl.get('data/'+str(signal_list[signal_index])) # fetch one signal from the file
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data = np.array(dataset)

    # plot
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(9,7))
    ax1.plot(np.linspace(0,60,6000),data[:,2],color='k',linewidth=1) # plot waveform
    ymin, ymax = ax1.get_ylim()
    ax1.vlines(dataset.attrs['p_arrival_sample']/100,ymin,ymax,color='b',linewidth=1.5, label='P-arrival') # plot p-wave arrival time
    ax1.vlines(dataset.attrs['s_arrival_sample']/100, ymin, ymax, color='r', linewidth=1.5, label='S-arrival') # plot s-wave arrival time
    ax1.vlines(dataset.attrs['coda_end_sample']/100, ymin, ymax, color='cyan', linewidth=1.5, label='Coda end')
    ax1.set_xlim([0,60])
    ax1.legend(loc='lower right',fontsize=10)
    ax1.set_ylabel('Amplitude (counts)')
    ax1.set_xlabel('Time (s)')
    im = ax2.specgram(data[:,2],Fs=100,NFFT=256,cmap='jet',vmin=-10,vmax=25); # plot spectrogram
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax3.psd(data[:,2],256,100,color='cornflowerblue') # plot power spectral density
    ax3.set_xlim([0,50])
    plt.savefig('waveform_spectrogram_plot.png',dpi=500)
    plt.tight_layout()
    plt.show()

    print('The p-wave for this waveform was picked by: ' + dataset.attrs['p_status'])
    print('The s-wave for this waveform was picked by: ' + dataset.attrs['s_status'])

# %%
waveform_spectrogram_plot(file_name,12000,df['trace_name'].to_list())