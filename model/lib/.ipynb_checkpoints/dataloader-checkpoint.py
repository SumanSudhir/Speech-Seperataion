import numpy as np
import pandas as pd
import torch

class DataLoader(object):

    def __init__(self, train=True, number_of_people=2, shuffle=True, batchsize=1, dataset_loc='', device='cpu'):

        self.dataset_loc = dataset_loc
        self.mix_audio_loc = self.dataset_loc + '/AV_model_database/mix/'
        self.single_audio_loc = self.dataset_loc + '/AV_model_database/single/'
        self.face_emb_loc = self.dataset_loc + '/face1022_emb_train/'
        self.device = device
        
        self.mean_array = np.load(self.dataset_loc+'/face_emb_mean.npy')
        self.std_array = np.load(self.dataset_loc+'/face_emb_std.npy')
        
        if train is True:
            self.data_file = pd.read_csv(self.dataset_loc + '/AV_model_database/AVdataset_train.txt', header=None)[0]
            self.shuffle = shuffle
            data_list = []
            for i, line in enumerate(self.data_file[:]):
                if 'single-000187.npy' in line.split():
                    data_list.append(i)
                    
            self.data_file = self.data_file[data_list]
            
        else:
            self.data_file = pd.read_csv(self.dataset_loc + '/AV_model_database/AVdataset_val.txt', header=None)[0]
            self.shuffle = shuffle
            data_list = []
            for i, line in enumerate(self.data_file[:]):
                if 'single-000187.npy' in line.split():
                    data_list.append(i)
                    
            self.data_file = self.data_file[data_list]
            
#         print(len(self.data_file))  
        self.batchsize = batchsize
        self.number_of_people = number_of_people

        if self.batchsize > self.data_file.shape[0]:
            self.batchsize = self.data_file.shape[0]

    def train_loader(self, batch_file_list):

        audio_input = np.zeros((self.batchsize, 298, 257, 2))
        video_input = np.zeros((self.batchsize, 75, 1, 1792, self.number_of_people))
        target_label = np.zeros((self.batchsize, 298, 257, 2, self.number_of_people))
#         print(batch_file_list)
        for i, line in enumerate(batch_file_list):
            line_list = line.split()
            mix_audio = line_list[0]
            single_audio_list = []
            face_emb_list = []

            for j in range(self.number_of_people):
                single_audio_list.append(line_list[j + 1])

            for j in range(self.number_of_people):
                face_emb_list.append(line_list[j + self.number_of_people + 1])

            audio_input[i] = np.load(self.mix_audio_loc + mix_audio)

            for j in range(self.number_of_people):
                video_input[i, :, :, :, j] = np.load(self.face_emb_loc + face_emb_list[j])
                video_input[i, :, :, :, j] -= self.mean_array 
                video_input[i, :, :, :, j] /= self.std_array 

            for j in range(self.number_of_people):
                target_label[i, :, :, :, j] = np.load(self.single_audio_loc + single_audio_list[j])
            
            
        return audio_input, video_input, target_label

    def __getitem__(self, index):
        
#         if self.shuffle is True:
#             self.data_file = self.data_file.sample(frac=1).reset_index(drop=True)
    
#         if (index + 1) * self.batchsize  > self.data_file.shape[0]:        
# #             index = 0
#             index = index%(self.data_file.shape[0]//self.batchsize)


        if self.shuffle is False:
            temp_data_list = self.data_file[index * self.batchsize: (index + 1) * self.batchsize]
        else:
            temp_data_list = self.data_file.sample(n=self.batchsize,replace=True)

        audio_input, video_input, target_label = self.train_loader(temp_data_list)

        
        audio_input = torch.from_numpy(audio_input.transpose(0, 3, 1, 2)).double().to(self.device)
        video_input = torch.from_numpy(video_input.transpose(0, 3, 1, 2, 4)).double().to(self.device)
        target_label = torch.from_numpy(target_label).double().to(self.device)
        return audio_input, video_input, target_label


# DataLoader_train = DataLoader(train=True, number_of_people=2, shuffle=False, batchsize=5, dataset_loc='../../data/dataset', device='cpu')
# audio, video, target = DataLoader_train[51]
# print(audio.shape, video.shape, target.shape)
# print(video[0,:,:,0,0].shape)
# print(video[0,:,:,0,0].min())
# print(video[0,:,:,0,0].max())
# print(video[0,:,:,0,0].mean())
# print(video[0,:,:,0,0].std())


