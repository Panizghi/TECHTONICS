'''
following script reads the metadata CSV files and extracts the waveform signal data from the corresponding HDF5 file.
The two types of images produced by this script are:
Waveform plot of signal
Spectrogram plot of sign

Normalized the color axis of the spectrograms to the range of -10 to 25 db/Hz 
Spectrograms created using NFFT of 256

'''

#docker image from kaggle: docker pull gcr.io/kaggle-images/python:v131

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel,delayed
import os

hd5_1 ="/kaggle/input/chunk1/chunk1.hdf5"
csv_file_1 = "/kaggle/input/chunk1/chunk1.csv"
hdf5_2 =  "/kaggle/input/stead-chunk-2/chunk2.hdf5"
csv_file_2 = "/kaggle/input/stead-chunk-2/chunk2.csv"

# read csv files into Pandas dataframes
chunk_1 = pd.read_csv(csv_file_1)
chunk_2 = pd.read_csv(csv_file_2)

# concatenate the two chunks into a single dataframe
full_csv = pd.concat([chunk_1,chunk_2])

# select chunk of data
chunk_name = full_csv

# select start of data rows you want to pull from that chunk
data_start = 0

# select end of data rows you want to pull from that chunk
data_end = 126500

# select interval you'd like to pull (smaller interval with more loops may run faster)
data_interval = 500

# select path to earthquake data chunk
eqpath = full_csv

img_save_path = '/kaggle/working/'

# Make images
eqlist = chunk_name['trace_name'].to_list()
eqlist = np.random.choice(eqlist,126500,replace=False) # turn on to get random sample of signals

starts = list(np.linspace(data_start,data_end-data_interval,int((data_end-data_start)/data_interval)))
ends = list(np.linspace(data_interval,data_end,int((data_end-data_start)/data_interval)))
set = str(chunk_name)
total_size = 0
count = 0

for n in range(0,len(starts)):
    traces = eqlist[int(starts[n]):int(ends[n])]
    path = eqpath
    count += 1
    
    def make_images(i):
        try:
            # retrieve selected waveforms from the hdf5 file
            dtfl = h5py.File(path, 'r')
            dataset = dtfl.get('/kaggle/input/'+str(traces[i]))
            # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
            data = np.array(dataset)
    
            # plot waveform and save as PNG
            fig, ax = plt.subplots(figsize=(3,2))
            ax.plot(np.linspace(0,60,6000),data[:,2],color='k',linewidth=1)
            ax.set_xlim([0,60])
            ax.axis('off')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.savefig(img_save_path+traces[i]+'.png',bbox_inches='tight',dpi=50)
            size = os.path.getsize(img_save_path+traces[i]+'.png')
            plt.close()

            # plot spectrogram and save as PNG
            fig, ax = plt.subplots(figsize=(3,2))
            ax.specgram(data[:,2], Fs=100, NFFT=256, cmap='gray', vmin=-10, vmax=25);
            ax.set_xlim([0,60])
            ax.axis('off')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            plt.savefig(img_save_path + traces[i] + '.png', bbox_inches='tight', transparent=True, pad_inches=0, dpi=50)
            size += os.path.getsize(img_save_path + traces[i] + '.png')
            plt.close()

            return size

        except:
            return 0


if __name__ == '__main__':
    # create images for selected data (runs in parallel using joblib)
    total_size = 0
    count = 0
    for n in range(len(starts)):
        traces = eqlist[int(starts[n]):int(ends[n])]
        path = eqpath
        count += 1
        with Parallel(n_jobs=-1) as parallel:
            sizes = parallel(delayed(make_images)(i, path, traces) for i in range(len(traces)))
        total_size += sum(sizes)