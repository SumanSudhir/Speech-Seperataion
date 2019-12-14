## evaluate the model and generate the prediction
from __future__ import print_function
import sys
sys.path.append('../lib')
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import numpy as np
import pickle
from model import AV_model
from dataloader import DataLoader
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import scipy.io.wavfile as wavfile
import utils
import pandas as pd
people_num = 2

use_cuda = torch.cuda.is_available()
print('gpu status =', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
# device  ='cpu'
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# PATH
model_path = 'checkpoint_dir/model_10.pth'
dir_path = 'Pred/'

if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

database_path = '../../data/dataset/AV_model_database/'
face_path = '../../data/dataset/face1022_emb_train'

mean_array = np.load('../../data/dataset/face_emb_mean.npy')
std_array = np.load('../../data/dataset/face_emb_std.npy')

# load data
# testfiles = []
# with open('../../data/dataset/AV_model_database/AVdataset_val.txt', 'r') as f:
#     testfiles = f.readlines()
    
testfiles = pd.read_csv('../../data/dataset/AV_model_database/AVdataset_train.txt', header=None)[0]
data_list = []
for i, line in enumerate(testfiles[:]):
    if 'single-000187.npy' in line.split():
        data_list.append(i)
testfiles = testfiles[data_list]

# testfiles = testfiles[:5]
print(testfiles.shape)

def parse_X_data(line,num_people=people_num,database_path=database_path,face_path=face_path):
    parts = line.split() # get each name of file for one testset
    mix_str = parts[0]
    name_list = mix_str.replace('.npy','')
    name_list = name_list.replace('mix-','',1)
    names = name_list.split('-')
    single_idxs = []
    single_list = [None]*num_people
    for i in range(num_people):
        single_idxs.append(names[i])
    file_path = database_path + 'mix/' + mix_str
    mix = np.load(file_path)
    mix = np.expand_dims(mix, axis=0)
    face_embs = np.zeros((1,75,1,1792,num_people))
    for i in range(num_people):
        face_embs[0,:,:,:,i] = np.load(face_path+"/%06d_face_emb.npy"%int(single_idxs[i]))
        face_embs[0,:,:,:,i] = (face_embs[0,:,:,:,i] - mean_array)/std_array
        single_list[i] = np.load(database_path + 'single/single-' + single_idxs[i]+'.npy')
        
    face_embs = torch.from_numpy(face_embs.transpose(0, 3, 1, 2, 4)).double().to(device)
    mix = torch.from_numpy(mix.transpose(0, 3, 1, 2)).double().to(device)

    return mix,single_idxs,face_embs,single_list

def mse(true_,pred_):
    return np.mean(np.square(np.subtract(true_,pred_)))


# predict data

net = AV_model(people_num)
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
net.load_state_dict(checkpoint['model_state_dict'])
net = net.double().to(device)
print('Model Loaded')
net.eval()

with torch.no_grad():
    for n,line in enumerate(testfiles[:2]):
        print(n+1)
        mix, single_idxs, face_embs, single_audio_npy_list = parse_X_data(line)
        print(single_idxs)
        mix_ = mix.cpu().detach().numpy().transpose(0,2,3,1).reshape(298,257,2)
        print(np.unique(mix.cpu().detach().numpy().transpose(0,2,3,1).reshape(298,257,2)))
        mix_1 = np.where(mix_>-1,mix_,0)
        mix_1 = np.where(mix_1<1,mix_1,0)
        print(np.count_nonzero(mix_1),mix_1.size)
        mix_1 = np.where(mix_>-0.5,mix_,0)
        mix_1 = np.where(mix_1<0.5,mix_1,0)
        print(np.count_nonzero(mix_1),mix_1.size)
        print(np.unique(face_embs.cpu().detach().numpy().transpose(0,2,3,1,4)[:,:,:,:,0]))

        seperated_speech_spectograms = net(mix, face_embs)
        prefix = "/"
        for idx in single_idxs:
            prefix += idx + "-"
        for i in range(people_num):
            seperated_speech_spectogram = seperated_speech_spectograms[0,:,:,:,i]
            assert seperated_speech_spectogram.shape == (298,257,2)
            seperated_speech_spectogram = seperated_speech_spectogram.cpu().detach().numpy()
            
            print('seperated_speech_spectogram:',np.unique(seperated_speech_spectogram))

            mag_plot = np.sqrt(seperated_speech_spectogram[:,:,0]*seperated_speech_spectogram[:,:,0] + seperated_speech_spectogram[:,:,1]*seperated_speech_spectogram[:,:,1])
            plt.imsave('mag_'+str(n)+'_'+str(i)+'.png',mag_plot)
        
            single_plot = np.sqrt(single_audio_npy_list[i][:,:,0]*single_audio_npy_list[i][:,:,0] + single_audio_npy_list[i][:,:,1]*single_audio_npy_list[i][:,:,1])
            plt.imsave('mag_'+str(n)+'_'+str(i)+'GT'+'.png',single_plot)
            
            T = utils.fast_istft(seperated_speech_spectogram,power=True)
            
            print(np.count_nonzero(seperated_speech_spectogram),seperated_speech_spectogram.size)
            mix_1 = np.where(seperated_speech_spectogram>-1,seperated_speech_spectogram,0)
            mix_1 = np.where(mix_1<1,mix_1,0)
            print(np.count_nonzero(mix_1),mix_1.size)
            mix_1 = np.where(seperated_speech_spectogram>-0.5,seperated_speech_spectogram,0)
            mix_1 = np.where(mix_1<0.5,mix_1,0)
            print(np.count_nonzero(mix_1),mix_1.size)
        
#             mix_ = mix.cpu().detach().numpy().transpose(0,2,3,1).reshape(298,257,2)/2
#             T = utils.fast_istft(mix_,power=True)

            filename = dir_path+prefix+single_idxs[i]+'.wav'
            wavfile.write(filename,16000,T)

