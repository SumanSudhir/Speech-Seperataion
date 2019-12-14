import librosa
import os
import numpy as np
import scipy.io.wavfile as wavfile
import argparse
import shutil

emb_dir = 'face1022_emb_train'

temp_array = np.zeros((75,1,1792))

for face in os.listdir(emb_dir)[:5]:
    if face.split('.')[-1] == 'npy':
        temp_npy = np.load(emb_dir+'/'+face)
        temp_array += temp_npy

mean_array = np.zeros((75,1,1792))
std_array = np.zeros((75,1,1792))

for i in range(1792):
    val = np.reshape(temp_array[:,:,i], -1)
    mean_array[:,:,i] = np.mean(val)
    std_array[:,:,i] = np.std(val)

np.save('face_emb_mean.npy', mean_array)
np.save('face_emb_std.npy', std_array)